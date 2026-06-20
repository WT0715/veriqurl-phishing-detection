# VeriqURL

**VeriqURL** is a lightweight URL-based phishing detection and awareness prototype. It allows a user to paste a suspicious URL, receive phishing predictions from two models, view a combined risk decision, and read evidence-based awareness guidance based on detected URL indicators.

Live demo: https://veriqurl-phishing-detection.streamlit.app/

---

## Project Purpose

Phishing links are commonly shared through email, SMS, messaging applications, and social media. Non-expert users may find it difficult to inspect the real domain, recognise suspicious URL wording, or understand why a link looks risky.

VeriqURL focuses on **URL-only analysis**. It does not visit the webpage, download HTML, execute JavaScript, inspect screenshots, or collect personal information. This keeps the system lightweight and suitable for fast single-URL checking.

---

## Key Features

- **Random Forest phishing prediction** using handcrafted lexical and structural URL features.
- **Character-level TCN phishing prediction** using encoded URL character sequences.
- **Combined decision** based on the average phishing probability from both models.
- **Three-level risk output**: Low Risk, Suspicious, and High Risk.
- **Evidence-based explanation layer** that identifies suspicious URL indicators.
- **Evidence-based awareness guidance** instead of generic recommendations.
- **Model and lightweight metrics dashboard** showing accuracy, precision, recall, F1-score, training time, testing time, per-sample latency, TES, IES, RTDE, model size, and parameter count.
- **Full pipeline notebook** for reproducibility and supervisor review.

---

## Why VeriqURL?

VeriqURL is not designed to replace enterprise phishing gateways, browser safe-browsing systems, or commercial threat-intelligence products. Instead, it is designed as a lightweight educational and awareness-support tool for users who want to check a suspicious URL before trusting it.

Its strengths are:

1. **Lightweight URL-only design**  
   The system analyses only the URL string, reducing feature-acquisition cost and avoiding webpage interaction.

2. **Dual-model comparison**  
   Random Forest provides a fast feature-based baseline, while TCN learns URL character-sequence patterns directly.

3. **Balanced detection and efficiency evaluation**  
   The project evaluates both classification performance and lightweight efficiency, instead of reporting accuracy only.

4. **Evidence-based awareness guidance**  
   The system explains suspicious URL indicators and provides user guidance based on known phishing patterns such as suspicious account/security words, unusual brand placement, HTTP usage, multiple hyphens, shorteners, IP-based URLs, and long or obfuscated paths.

---

## Detection Logic

VeriqURL follows this detection flow:

1. **User submits one URL** through the Streamlit interface.
2. **Input validation** checks whether the input is empty or appears to contain more than one URL.
3. **Random Forest prediction** extracts handcrafted URL features such as:
   - URL length
   - domain length
   - path length
   - query length
   - digit count
   - dot count
   - hyphen count
   - slash count
   - HTTPS usage
   - IP address usage
   - suspicious keyword count
4. **TCN prediction** converts the URL into a fixed-length character sequence and predicts phishing probability using a character-level Temporal Convolutional Network.
5. **Combined decision** averages the phishing probabilities from Random Forest and TCN.
6. **Risk mapping** converts the final probability into a user-friendly risk level.
7. **Explanation and awareness guidance** are generated using a shared rule-based explanation layer.

---

## Risk Level Mapping

| Risk Level | Probability Range | Meaning |
|---|---:|---|
| Low Risk | `p < 0.50` | The model probability is below the phishing decision boundary. |
| Suspicious | `0.50 <= p < 0.80` | The URL is predicted as phishing, but not high-confidence enough for the strongest warning. |
| High Risk | `p >= 0.80` | The URL is a high-confidence phishing prediction. |

The `0.80` High Risk threshold was selected using validation-set threshold testing. Candidate thresholds of `0.60`, `0.70`, `0.80`, and `0.90` were compared based on High Risk precision, High Risk recall, false positives, and Suspicious count.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9960 | 0.9987 | 0.9920 | 0.9953 |
| TCN | 0.9985 | 0.9999 | 0.9966 | 0.9983 |

The TCN model achieved slightly stronger classification performance, while Random Forest remained highly competitive as a lightweight baseline.

---

## Lightweight Efficiency

| Model | Training Time (s) | Testing Time (s) | Latency (ms) | TES | IES | RTDE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 3.4625 | 0.0845 | 0.0024 | 0.2877 | 11.7933 | 416.3743 |
| TCN | 6433.3831 | 23.9394 | 0.6781 | 0.0002 | 0.0417 | 1.4726 |

Random Forest is more lightweight in terms of training time, testing time, and per-sample latency. TCN is less lightweight during training and batch testing, but it still has acceptable real-time latency for single-URL analysis.

---

## Project Structure

```text
Phishing Detection/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
│  ├─ random_forest_model.pkl
│  └─ tcn_model.pt
├─ notebooks/
│  └─ VeriqURL_Full_Pipeline.ipynb
├─ results/
│  ├─ model_comparison.csv
│  ├─ risk_threshold_validation.csv
│  └─ risk_threshold_distribution.csv
└─ src/
   ├─ preprocess.py
   ├─ split_data.py
   ├─ feature_extraction.py
   ├─ train_rf.py
   ├─ prepare_tcn_data.py
   ├─ train_tcn.py
   ├─ evaluate_tcn.py
   ├─ compare_models.py
   ├─ validate_risk_thresholds.py
   ├─ explanation_rules.py
   └─ predict_compare.py
```

---

## Run Locally

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Run the Full Pipeline Notebook

The notebook is located at:

```text
notebooks/VeriqURL_Full_Pipeline.ipynb
```

It documents the full project pipeline:

1. Dataset loading and cleaning
2. Train / validation / test split
3. Random Forest feature extraction
4. Random Forest training and evaluation
5. TCN character vocabulary and URL encoding
6. TCN architecture
7. TCN training summary
8. Model comparison and lightweight metrics
9. Risk threshold validation
10. Evidence-based explanation layer
11. Case study prediction

The notebook is mainly for reproducibility and explanation. The TCN training section should not be rerun unnecessarily because training is time-consuming.

---

## Example Case Study URLs

```text
https://www.google.com
http://secure-paypal-login-verification.com/account/verify
https://update-service-login.example.org
```

These examples demonstrate Low Risk, High Risk, and Suspicious/High Risk style outputs.

---

## Limitations

- The system only analyses the URL string.
- It does not inspect webpage content, HTML, JavaScript, screenshots, or visual similarity.
- It may miss phishing websites that use normal-looking URLs but malicious page content.
- The current evaluation is based on an internal train-validation-test split of the selected dataset.
- External cross-dataset robustness testing is not included in the final prototype.
- The tool is an educational and awareness-support prototype, not a replacement for enterprise security products.

---

## Future Work

- Add external dataset evaluation.
- Add optional webpage-content analysis while preserving lightweight mode.
- Add session-only URL history for usability testing.
- Add confidence calibration and model disagreement warnings.
- Extend the prototype into a browser extension or email-system integration.

---

## Safety and Privacy Boundary

VeriqURL does not ask users to enter usernames, passwords, banking details, or personal information. It only analyses the submitted URL string.
