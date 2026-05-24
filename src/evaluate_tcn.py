import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_DIR = "data/processed/tcn"
MODEL_PATH = "models/tcn_model.pt"
RESULTS_DIR = "results"

VOCAB_PATH = os.path.join(DATA_DIR, "char_vocab.json")
TEST_X_PATH = os.path.join(DATA_DIR, "test_X.npy")
TEST_Y_PATH = os.path.join(DATA_DIR, "test_y.npy")

BATCH_SIZE = 128
EMBED_DIM = 32
NUM_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2

class URLDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]
    

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TCNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_size, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.tcn = nn.Sequential(
            TemporalBlock(embed_dim, num_channels, kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(num_channels, num_channels, kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(num_channels, num_channels, kernel_size, dilation=4, dropout=dropout)
        )

        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)

        # Global max pooling:
        # use the strongest learned URL pattern found anywhere in the sequence
        x = torch.max(x, dim=2).values

        x = self.fc(x)
        return x.squeeze(1)
    

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    vocab_size = len(char_to_idx)

    test_dataset = URLDataset(TEST_X_PATH, TEST_Y_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TCNClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    num_parameters = sum(p.numel() for p in model.parameters())
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    all_probs = []
    all_preds = []
    all_labels = []

    testing_start_time = time.time()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    testing_time_s = time.time() - testing_start_time
    per_sample_time_s = testing_time_s / len(test_dataset)
    per_sample_latency_ms = per_sample_time_s * 1000

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    pred_positive_rate = np.mean(all_preds)

    training_summary_path = os.path.join(RESULTS_DIR, "tcn_training_summary.csv")
    if os.path.exists(training_summary_path):
        training_summary_df = pd.read_csv(training_summary_path)
        training_time_s = float(training_summary_df.loc[0, "training_time_s"])
    else:
        training_time_s = None

    tes = accuracy / training_time_s if training_time_s and training_time_s > 0 else None
    ies = accuracy / testing_time_s if testing_time_s > 0 else None
    rtde = accuracy / per_sample_latency_ms if per_sample_latency_ms > 0 else None

    print("\n===== TCN Test Results =====")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    print(f"Pred Positive Rate:   {pred_positive_rate:.4f}")

    print("\n===== Confusion Matrix =====")
    print(cm)

    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=["benign", "phishing"], zero_division=0))

    metrics_df = pd.DataFrame([{
        "model": "TCN",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pred_positive_rate": pred_positive_rate,
        "training_time_s": training_time_s,
        "testing_time_s": testing_time_s,
        "per_sample_time_s": per_sample_time_s,
        "per_sample_latency_ms": per_sample_latency_ms,
        "tes": tes,
        "ies": ies,
        "rtde": rtde,
        "model_size_mb": model_size_mb,
        "num_parameters": num_parameters
    }])

    metrics_df.to_csv(os.path.join(RESULTS_DIR, "tcn_metrics.csv"), index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["actual_benign", "actual_phishing"],
        columns=["predicted_benign", "predicted_phishing"]
    )

    cm_df.to_csv(os.path.join(RESULTS_DIR, "tcn_confusion_matrix.csv"))

    print(f"Training Time:        {training_time_s:.4f} seconds" if training_time_s else "Training Time:        Not available")
    print(f"Testing Time:         {testing_time_s:.6f} seconds")
    print(f"Per-sample Time:      {per_sample_time_s:.8f} seconds")
    print(f"Per-sample Latency:   {per_sample_latency_ms:.6f} ms")
    print(f"TES:                  {tes:.6f}" if tes else "TES:                  Not available")
    print(f"IES:                  {ies:.6f}")
    print(f"RTDE:                 {rtde:.6f}")
    print(f"Model Size:           {model_size_mb:.4f} MB")
    print(f"Num Parameters:       {num_parameters}")

    print("\nSaved:")
    print("results/tcn_metrics.csv")
    print("results/tcn_confusion_matrix.csv")


if __name__ == "__main__":
    main()