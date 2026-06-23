# intrusion

Here is a professional, clean, and comprehensive `README.md` written in English for your Master's thesis project. It is structured beautifully for GitHub, complete with technical details extracted from your code, model architecture, and validation metrics.

---

# Network Intrusion Detection System (NIDS) using Deep Learning

An advanced, high-performance Network Intrusion Detection System (NIDS) built to detect cyberattacks in financial environments and complex corporate networks. Leveraging the industry-standard **CICIDS2017 dataset**, this project uses a multi-layered Long Short-Term Memory (**LSTM**) network paired with a high-speed, mathematical **Pure Inference NumPy Engine** for real-time traffic monitoring.

---

## 🚀 Key Features

* **Advanced Feature Selection**: Utilizes a Random Forest algorithm to reduce feature complexity down to the **top 15 dominant network metrics**, ensuring speed without compromising precision.


* **3D Temporal Sequence Mapping**: Evaluates active network connections using rolling time-series windows ($T=5$) to capture the complex sequential behaviors of modern exploits.


* **RAM-Optimized Training Pipeline**: Implements a balanced online filtering system to handle massive, multi-gigabyte PCAP-derived CSV logs safely on standard hardware.


* **Ultra-Fast NumPy Inference Engine**: Bypasses heavy deep learning frameworks during deployment by migrating trained layers into pure matrix mathematics, delivering near-zero latency.



---
## Dataset

The model is trained and evaluated using the **CICIDS2017** dataset provided by the Canadian Institute for Cybersecurity (UNB).
*   **Official Dataset Link:** [UNB CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

## 📊 Neural Network Architecture

The deep learning model is built on top of the modern **Keras 3 architecture** and is optimized utilizing an `Adam` optimizer alongside batch normalization and dropout layers to maximize stability.

```
┌────────────────────────────────────────────────────────┐
│      Input Layer: shape=(None, 5, 15)                  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│      LSTM Layer 1 (64 Units) -> Tanh/Sigmoid           │
├────────────────────────────────────────────────────────┤
│      Batch Normalization Layer 1 (Epsilon=0.001)       │
├────────────────────────────────────────────────────────┤
│      Dropout Layer 1 (Rate=0.3)                        │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│      LSTM Layer 2 (32 Units, return_sequences=False)   │
├────────────────────────────────────────────────────────┤
│      Batch Normalization Layer 2 (Epsilon=0.001)       │
├────────────────────────────────────────────────────────┤
│      Dropout Layer 2 (Rate=0.2)                        │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│      Dense Softmax Output Layer (15 Threat Classes)     │
└────────────────────────────────────────────────────────┘

```

---

## 📈 Evaluation Metrics & Results

The system achieves a stellar final evaluation, effectively handling severe class imbalances (e.g., matching rare or low-volume attacks accurately).

### Model Performance Summary

* **Overall Test Accuracy**: **82.77%**

* **Dominant Attack Performance**: Reached a flawless **1.00 F1-Score** on devastating structural attacks such as `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP`, `DoS attacks-Hulk`, and `SSH-Bruteforce`.



### Performance Visualization

The detailed visual validation metrics are saved directly in the project root:

1. **`chart_1_confusion_matrix.jpg`**: A comprehensive cross-class evaluation map exposing exact prediction distributions across all 15 active threat classifications.


2. **`chart_2_f1_scores_bars.jpg`**: Bar chart detailing individual $F_1$-scores per class, highlighting robust classification baseline performance even on complex, rare attack variations.



---

## 📂 Core Project Artifacts

* **`scaler.pkl`**: Standardized feature scaler used to normalize network parameters.


* **`label_encoder.pkl`**: String-to-index tracking matrix containing the 15 verified classes.


* **`feature_names.pkl`**: Reference tracking list of the top 15 selected network variables.


* **`cicids_numpy_weights.pkl`**: Saved raw tensor bias/kernel matrices powering the standalone prediction engine.



---

## 🛠️ Monitored Features List

The model dynamically monitors the following **15 extracted features** to build a continuous threat model:

| # | Feature Name | # | Feature Name |
| --- | --- | --- | --- |
| **1** | `Init Fwd Win Byts`<br> | **9** | `Bwd Pkts/s`<br> |
| **2** | `Fwd Seg Size Min`<br> | **10** | `Fwd IAT Tot`<br> |
| **3** | `Fwd Header Len`<br> | **11** | `Fwd Pkts/s`<br> |
| **4** | `Flow Duration`<br> | **12** | `TotLen Fwd Pkts`<br> |
| **5** | `Flow IAT Mean`<br> | **13** | `Fwd IAT Min`<br> |
| **6** | `Flow IAT Max`<br> | **14** | `Subflow Fwd Pkts`<br> |
| **7** | `Flow Pkts/s`<br> | **15** | `Init Bwd Win Byts`<br> |
| **8** | `Flow IAT Min`<br> |  |  |

---

## 💻 Usage & Deployment

### Training the Model

To run the automated training, data balancing, and feature extraction pipeline:

```bash
python train_pipeline.py

```

### Running the Live Production Simulation

To run the lightweight network tracking simulation powered exclusively by the standalone NumPy engine:

```bash
python live_inference_simulation.py

```

---

## 🎓 Academic Affiliation

* **Author**: Deyan Denev
* **University**: University "Prof. Dr. Assen Zlatarov" – Burgas
* **Faculty**: Faculty of Technical Sciences, Department of Computer Systems and Technologies
* **Master's Program**: Cybersecurity: Technologies in the Financial Sphere
* **Academic Advisor**: Assoc. Prof. Dr. Ivaylo Mihaylov

