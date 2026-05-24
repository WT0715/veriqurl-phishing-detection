import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_DIR = "data/processed/tcn"
MODEL_DIR = "models"
RESULTS_DIR = "results"

VOCAB_PATH = os.path.join(DATA_DIR, "char_vocab.json")
TRAIN_X_PATH = os.path.join(DATA_DIR, "train_X.npy")
TRAIN_Y_PATH = os.path.join(DATA_DIR, "train_y.npy")
VAL_X_PATH = os.path.join(DATA_DIR, "val_X.npy")
VAL_Y_PATH = os.path.join(DATA_DIR, "val_y.npy")

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0005
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
        y = torch.tensor(self.y[idx], dtype=torch.float32)
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
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)          # (batch, embed_dim, seq_len)
        x = self.tcn(x)                # (batch, num_channels, seq_len)
        
        # Global max pooling over the sequence length.
        # This lets the model use the strongest suspicious patterns
        # found anywhere in the URL, instead of only using the final position.
        x = torch.max(x, dim=2).values  # (batch, num_channels)
        
        x = self.fc(x)                # (batch, 1)
        return x.squeeze(1) 
    
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    pred_positive_rate = np.mean(all_preds)

    return accuracy, precision, recall, f1, pred_positive_rate


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    vocab_size = len(char_to_idx)

    train_dataset = URLDataset(TRAIN_X_PATH, TRAIN_Y_PATH)
    val_dataset = URLDataset(VAL_X_PATH, VAL_Y_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TCNClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)

    train_labels = np.load(TRAIN_Y_PATH)
    num_neg = (train_labels == 0).sum()
    num_pos = (train_labels == 1).sum()

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"num_neg: {num_neg}, num_pos: {num_pos}")
    print(f"Using pos_weight: {pos_weight.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    history = []
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_accuracy, val_precision, val_recall, val_f1, pred_positive_rate = evaluate(model, val_loader, device)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss:        {avg_loss:.4f}")
        print(f"Val Accuracy:      {val_accuracy:.4f}")
        print(f"Val Precision:     {val_precision:.4f}")
        print(f"Val Recall:        {val_recall:.4f}")
        print(f"Val F1-score:      {val_f1:.4f}")
        print(f"Pred Positive Rate:{pred_positive_rate:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tcn_model.pt"))
            print("Best model saved.")

    training_time_s = time.time() - training_start_time
    training_summary_df = pd.DataFrame([{
        "model": "TCN",
        "training_time_s": training_time_s,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "embed_dim": EMBED_DIM,
        "num_channels": NUM_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "best_val_f1": best_f1
    }])
    training_summary_df.to_csv(
        os.path.join(RESULTS_DIR, "tcn_training_summary.csv"),
        index=False
    )
    print(f"TCN training time: {training_time_s:.4f} seconds")
    print("Training summary saved to results/tcn_training_summary.csv")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_DIR, "tcn_training_history.csv"), index=False)
    print("Training completed. History saved.")


if __name__ == "__main__":
    main()
    


