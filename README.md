# 🛡️ FraudGuard AI — Financial Fraud Detection System

A full end-to-end machine learning pipeline for detecting financial fraud, with an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
fraud_detection/
│
├── app.py                  ← Main Streamlit Dashboard (5 pages)
├── data_generator.py       ← Synthetic transaction data generator + streaming simulator
├── models.py               ← All ML models + pipeline
├── requirements.txt        ← Python dependencies
├── models/                 ← Saved model artifacts (auto-created on first run)
│   └── fraud_model.joblib
└── README.md
```

---

## 🚀 Quick Start

### 1. Create & activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

> **Note:** First launch trains 5 models on 8,000 synthetic transactions — takes ~60 seconds.
> Subsequent runs load the saved model instantly.

---

## 🤖 Models Implemented

| Model | Type | Description |
|---|---|---|
| **Isolation Forest** | Unsupervised | Isolates anomalies via random partitioning |
| **One-Class SVM** | Unsupervised | Learns boundary of normal transaction space |
| **Autoencoder (MLP)** | Unsupervised | Reconstruction error = anomaly score |
| **Random Forest** | Supervised | Ensemble of decision trees with balanced class weights |
| **Gradient Boosting** | Supervised | Sequential boosting for high precision |
| **Ensemble** | Hybrid | Weighted average of all 5 models (best performance) |

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview Dashboard** | KPIs, time series, risk distribution, fraud by category |
| **Transaction Analysis** | Filterable transaction table + feature distributions |
| **Model Performance** | AUC, F1, confusion matrix, feature importance |
| **Live Detection** | Simulated Kafka streaming with real-time fraud alerts |
| **Single Transaction** | Manual input checker with fraud score gauge |

---

## 🔧 Key Features

- **ETL Pipeline**: Synthetic data generation simulating real bank transactions
- **Multi-Model Ensemble**: 5 models voting for highest accuracy
- **Real-Time Simulation**: Kafka-style streaming transaction scoring
- **Interactive Filters**: Risk level, merchant category, fraud score threshold
- **Feature Engineering**: Utilization ratio, amount-to-limit, distance, burst detection
- **Model Persistence**: Auto-saves trained models using joblib

---

## 📈 Synthetic Dataset Features

| Feature | Description |
|---|---|
| `amount` | Transaction amount (₹) |
| `merchant_category` | Type of merchant |
| `customer_age` | Age of card holder |
| `account_age_days` | How old is the account |
| `credit_limit` | Card credit limit |
| `available_balance` | Remaining balance |
| `is_international` | Whether transaction is abroad |
| `hour_of_day` | Time of transaction |
| `distance_from_home_km` | Distance from registered address |
| `num_transactions_last_24h` | Burst activity indicator |
| `utilization_ratio` | Balance used / credit limit |
| `amount_to_limit_ratio` | Amount as fraction of limit |

---

## 🧠 Fraud Heuristics (used in data generation)

Normal transactions:
- Business hours (6 AM – 10 PM)
- Domestic, low distance from home
- Low burst activity
- Moderate balance utilization

Fraudulent transactions:
- Late-night / early morning hours
- International, high distance
- High burst activity (10–40 tx/24h)
- Low available balance
- Very high or micro amounts

---

## 📦 Dependencies

```
streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, seaborn, imbalanced-learn, joblib, scipy
```

---

## 👨‍💻 Tech Stack

- **Data**: pandas, numpy
- **ML**: scikit-learn (Isolation Forest, One-Class SVM, MLP, Random Forest, GBM)
- **Visualization**: Plotly, Streamlit
- **Persistence**: joblib
- **Streaming Simulation**: Python generator (mimics Apache Kafka consumer)
