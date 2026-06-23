
# 🛡️ Deep Learning Network Intrusion Detection System (NIDS)

Refined, high-throughput network anomaly classification pipeline engineered specifically for high-risk corporate and financial sectors. This system employs an advanced recurrent architecture optimized for processing sequential connection traffic, paired with an ultra-lightweight math engine for production deployment.

---

## 💻 Tech Stack & Ecosystem

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🚀 Architectural Highlights

* **Dimensionality Reduction**: Eliminates noisy, co-linear network markers down to the **top 15 dominant metrics** via algorithmic tree-based selection.
* **Temporal Windows ($T=5$)**: Maps packet sequences into time-series tensors to intercept continuous multi-vector exploit behavior.
* **Framework-Free Execution**: Migrates deep tensor maps directly into hardcoded `NumPy` matrices, completely cutting runtime dependencies and eliminating framework-level latency.

---

## 📊 Neural Network Core

The system is powered by a multi-layered Long Short-Term Memory (**LSTM**) network built natively on the updated **Keras 3** specifications.


```

[Input Layer] ──► Shape: (None, 5, 15) Network Tensors
│
▼
[LSTM Layer 1] ──► 64 Memory Units (return_sequences=True)
│  ├── Batch Normalization (ε=0.001)
│  └── Dropout Regularization (Rate=0.3)
▼
[LSTM Layer 2] ──► 32 Memory Units (return_sequences=False)
│  ├── Batch Normalization (ε=0.001)
│  └── Dropout Regularization (Rate=0.2)
▼
[Dense Output] ──► 15 Softmax Channels (Probability Distribution)

```

> [!NOTE]
> The classification head maps outputs to **15 distinct target classes**, tracking everything from benign baseline operations to automated denial-of-service matrices and unauthorized protocol exploits.

---

## 📈 Performance & Visual Analytics

The neural execution pipeline successfully resolves extreme class imbalances inherently found in raw multi-gigabyte PCAP logs.

* **Global Classification Accuracy**: **82.77%**
* **Target Attack Precision**: Secured an optimal **1.00 F1-Score** on high-volume attack structural variants (`DDOS attack-HOIC`, `DDOS attack-LOIC-UDP`, `DoS attacks-Hulk`, and `SSH-Bruteforce`).

### 🖼️ Diagnostic Deliverables

Upon completion of execution, the pipeline auto-generates comprehensive visual telemetry graphs in the workspace:

1. **`complete_evaluation_metrics.png`** — Core diagnostic graphic combining the unified multi-class confusion matrix and cross-metric graphs.
2. **`chart_1_confusion_matrix.jpg`** — Expanded breakdown tracking precise false-positive to false-negative boundaries.
3. **`chart_2_f1_scores_bars.jpg`** — Granular evaluation comparison across rare or low-volume threats.

---

## 💾 Dataset Profile

The architecture uses the benchmark **CICIDS2017** dataset distributed by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB).

🔗 **Official Repository Link:** [UNB CICIDS2017 Source Material](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## 🛠️ Dynamic Metric Tracking Matrix

The monitoring array parses and transforms **15 exact feature vectors** from real-time network streams:

| Vector ID | Inspected Feature Parameter | Vector ID | Inspected Feature Parameter |
| :---: | :--- | :---: | :--- |
| `01` | **Init Fwd Win Byts** | `09` | **Bwd Pkts/s** |
| `02` | **Fwd Seg Size Min** | `10` | **Fwd IAT Tot** |
| `03` | **Fwd Header Len** | `11` | **Fwd Pkts/s** |
| `04` | **Flow Duration** | `12` | **TotLen Fwd Pkts** |
| `05` | **Flow IAT Mean** | `13` | **Fwd IAT Min** |
| `06` | **Flow IAT Max** | `14` | **Subflow Fwd Pkts** |
| `07` | **Flow Pkts/s** | `15` | **Init Bwd Win Byts** |
| `08` | **Flow IAT Min** | | |

---

## 📂 Production & Model Artifacts

* 📦 `scaler.pkl` — Standardized standard variance parameter configuration.
* 📦 `label_encoder.pkl` — Vector-to-string target tracking encoder mapping 15 classes.
* 📦 `feature_names.pkl` — Reference index containing structural column tracking definitions.
* 📦 `cicids_numpy_weights.pkl` — Extracted dense matrix array weight pairs used by the mathematical engine.
* 📦 `cicids_intrusion_model.keras` — Complete native Keras 3 model configuration and weight parameters.

---

## ⚙️ Operation & Deployment Guide

### 1. Execute Training & Feature Pipeline
To perform standard balancing, target vector selection, model training, and diagnostic rendering:

```bash
# Using standard script interface
python int.py

```

> [!TIP]
> Alternatively, you can run the entire model exploration, debugging blocks, and iterative training loops interactively through the Jupyter workspace inside **`LSTM.ipynb`**.

### 2. High-Speed Production Simulation (Pure NumPy)

To launch the high-speed execution mode driven solely by pure linear algebra matrices, running independently of TensorFlow:

```bash
python real.py

```

### 3. Interactive Thesis Live Presentation Mode

For a visually smoothed terminal stream optimized for academic presentation and real-time defense demonstration panels:

```bash
python live_demo.py

```

---

## 🎓 Academic Affiliation

* **Author:** Deyan Denev
* **University:** University "Prof. Dr. Assen Zlatarov" – Burgas
* **Faculty:** Faculty of Technical Sciences, Department of Computer Systems and Technologies
* **Master's Program:** Cybersecurity: Technologies in the Financial Sphere
* **Academic Advisor:** Assoc. Prof. Dr. Ivaylo Mihaylov
