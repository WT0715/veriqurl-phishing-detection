# Case Study Results: RF and TCN URL Prediction

This section presents three case studies to demonstrate how the proposed phishing detection system performs on different types of URLs. The system compares the predictions from the Random Forest baseline model and the character-level TCN model. In addition, a shared explanation layer is used to generate user-friendly explanations, recommendations, and awareness tips.

---

## Case 1: Benign URL

**Input URL:**
https://www.google.com


**Prediction Results:**
| Model         | Predicted Label | Benign Probability | Phishing Probability | Risk Level |
| ------------- | --------------: | -----------------: | -------------------: | ---------- |
| Random Forest |          benign |             0.9978 |               0.0022 | Low Risk   |
| TCN           |          benign |             0.9986 |               0.0014 | Low Risk   |


**Combined Decision:**
| Item                         | Result   |
| ---------------------------- | -------- |
| Models Agree                 | Yes      |
| Average Phishing Probability | 0.0018   |
| Final Label                  | benign   |
| Final Risk Level             | Low Risk |


**Explanation:**
No strong suspicious lexical pattern was detected from the extracted URL indicators.


**Discussion:**
Both Random Forest and TCN classified the URL as benign with very low phishing probabilities. This is expected because the URL is short, uses HTTPS, does not contain suspicious keywords, and has a simple domain structure. This case shows that the system does not over-flag a common legitimate URL.


## Case 2: Obvious Phishing URL

**Input URL:**
http://secure-paypal-login-verification.com/account/verify


**Prediction Results:**
| Model         | Predicted Label | Benign Probability | Phishing Probability | Risk Level |
| ------------- | --------------: | -----------------: | -------------------: | ---------- |
| Random Forest |        phishing |             0.0000 |               1.0000 | High Risk  |
| TCN           |        phishing |             0.0000 |               1.0000 | High Risk  |


**Combined Decision:**
| Item                         | Result    |
| ---------------------------- | --------- |
| Models Agree                 | Yes       |
| Average Phishing Probability | 1.0000    |
| Final Label                  | phishing  |
| Final Risk Level             | High Risk |


**Explanation:**
The system flagged this URL because:
1. The URL uses HTTP instead of HTTPS.
2. The URL contains brand-related words in an unusual domain structure: paypal.
3. The URL contains suspicious or suggestive words such as account, login, paypal, secure, verification, and verify.
4. The domain contains multiple hyphens, which may indicate brand imitation or obfuscation.


**Discussion:**
Both models classified the URL as phishing with maximum phishing probability. This URL contains multiple common phishing indicators, including brand impersonation, account-related keywords, verification-related keywords, HTTP usage, and multiple hyphens. The shared explanation layer helps translate these technical indicators into user-friendly reasons.


## Case 3: Suspicious / Borderline URL

**Input URL:**
https://update-service-login.example.org


**Prediction Results:**
| Model         | Predicted Label | Benign Probability | Phishing Probability | Risk Level |
| ------------- | --------------: | -----------------: | -------------------: | ---------- |
| Random Forest |        phishing |             0.2616 |               0.7384 | Suspicious |
| TCN           |        phishing |             0.0000 |               1.0000 | High Risk  |


**Combined Decision:**
| Item                         | Result    |
| ---------------------------- | --------- |
| Models Agree                 | Yes       |
| Average Phishing Probability | 0.8692    |
| Final Label                  | phishing  |
| Final Risk Level             | High Risk |


**Explanation:**
The system flagged this URL because:
1. The URL contains suspicious or suggestive words: login, update.
2. The domain contains multiple hyphens, which may indicate brand imitation or obfuscation.


**Discussion:**
This case is useful because it shows the difference between Random Forest and TCN. The Random Forest model classified the URL as phishing but assigned a lower phishing probability compared to TCN, resulting in a Suspicious risk level. In contrast, the TCN model assigned a very high phishing probability.

This suggests that the character-level TCN is more sensitive to suspicious token combinations such as update-service-login. Unlike Random Forest, which relies on handcrafted lexical features, the TCN learns character-level sequence patterns from the URL. Therefore, it may identify suspicious combinations more strongly even when the URL does not contain an obvious brand name or IP address.

---

## Summary of Case Study Findings

| Case | URL Type | Random Forest Result | TCN Result | Combined Result |
|---|---|---|---|---|
| Case 1 | Benign URL | Low Risk | Low Risk | Low Risk |
| Case 2 | Obvious Phishing URL | High Risk | High Risk | High Risk |
| Case 3 | Suspicious / Borderline URL | Suspicious | High Risk | High Risk |

The case studies show that both models can correctly identify a common benign URL and an obvious phishing URL. For the suspicious/borderline URL, both models predicted phishing, but the TCN assigned a higher phishing probability than Random Forest. This suggests that the character-level TCN is more sensitive to suspicious token combinations in the URL sequence, while Random Forest provides a more interpretable and lightweight baseline based on handcrafted lexical features.

---

## Test Set Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9960 | 0.9987 | 0.9920 | 0.9953 |
| TCN           | 0.9985 | 0.9999 | 0.9966 | 0.9983 |

The TCN achieved slightly higher accuracy, recall, and F1-score than Random Forest on the test set. This indicates that the character-level TCN can capture sequential URL patterns that may not be fully represented by handcrafted lexical features. However, the Random Forest model still achieved very strong performance and remains useful as a fast and interpretable baseline.

---

## Lightweight Efficiency Comparison

| Model | Training Time (s) | Testing Time (s) | Per-sample Latency (ms) | TES | IES | RTDE | Model Size (MB) | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 3.4625 | 0.0845 | 0.0024 | 0.2877 | 11.7933 | 416.3743 | N/A | N/A |
| TCN | 6433.3831 | 23.9394 | 0.6781 | 0.0002 | 0.0417 | 1.4726 | 0.2829 | 72,449 |

### Lightweight Metrics Used

The lightweight evaluation uses three efficiency metrics:

- **TES (Training Efficiency Score)** = Accuracy / Training Time
- **IES (Inference Efficiency Score)** = Accuracy / Testing Time
- **RTDE (Real-time Detection Efficiency)** = Accuracy / Per-sample Latency

### Discussion

Random Forest achieved much higher TES, IES, and RTDE values compared with TCN. This is because Random Forest required significantly less training time and had much lower per-sample latency. Therefore, Random Forest is the more lightweight model in terms of computational efficiency.

The TCN required a much longer training time, but it achieved the highest detection performance. Its model size is only 0.2829 MB and it has 72,449 parameters, which shows that the trained model is still small. Its per-sample latency is 0.6781 ms, which is still suitable for real-time URL detection. Therefore, TCN is lightweight enough for inference and deployment, but it is not as lightweight as Random Forest during training.

Overall, Random Forest is the stronger lightweight baseline, while TCN provides better detection performance with acceptable real-time inference latency.

---

## Overall Discussion

The results show a trade-off between detection performance and computational efficiency. Random Forest is faster to train and infer, making it suitable for lightweight deployment and baseline comparison. TCN achieves better detection accuracy, recall, and F1-score by learning character-level URL patterns, but requires significantly longer training time.

The shared explanation layer improves the usability of both models by converting technical URL indicators into user-friendly explanations, recommendations, and awareness tips. This supports the awareness objective of the system, as users are not only shown whether a URL is risky, but also why it may be risky.