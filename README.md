# Time Series Forecasting using LSTM and Transformer

This project demonstrates **time series forecasting** using two popular deep learning architectures:

* **LSTM (Long Short-Term Memory)**
* **Transformer (Encoder-only)**

The goal is to compare how these models perform on a **univariate time series forecasting task** using real-world public transit data.

---

## 📌 Project Objective

Given the previous **N days** of data, predict the **next day’s value**.

We use the **Chicago Transit Authority (CTA) daily rides dataset** and forecast the next-day total rides.

---

## 📊 Dataset

* **Source:** City of Chicago Open Data Portal
* **Data:** Daily bus + rail passenger counts
* **Time Range Used:** 2001 – 2019 (post-COVID data removed)
* **Target Variable:** `total_rides`

---

## 🧠 Models Used

### 1. LSTM Model

* Captures temporal dependencies using gated recurrent units
* Suitable for sequential and time-dependent data

### 2. Transformer Model

* Encoder-only Transformer
* Uses self-attention to capture temporal relationships
* Lightweight architecture for fair comparison

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **PyTorch**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **scikit-learn**

---

## 📂 Project Structure

```text
lstm_transformer_time_series_project/
│
├── timeseries_project.py   # Main training & evaluation script
├── README.md               # Project documentation
└── venv/                   # Virtual environment (not committed)
```

---

## 🛠️ Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run

```bash
python timeseries_project.py
```

The script will:

1. Download the dataset
2. Preprocess the time series
3. Train LSTM and Transformer models
4. Evaluate performance using RMSE and MAE

---

## 📈 Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

These metrics compare predicted vs actual passenger counts.

---

## 🧪 Results (Sample)

```text
LSTM RMSE ≈ 1.35M | MAE ≈ 1.30M
Transformer RMSE ≈ 1.35M | MAE ≈ 1.30M
```

Both models perform similarly for short-term forecasting on this dataset.

---

## 🧠 Key Learnings

* LSTMs and Transformers can perform similarly on simple univariate time series
* Transformers may show advantages for longer forecasting horizons
* Proper sequence preparation is crucial for deep learning models

---

## 🚀 Future Improvements

* Multi-step forecasting (predict next 7 / 30 days)
* Multivariate inputs (weather, holidays, events)
* Hyperparameter tuning
* GPU acceleration

---

## 👨‍💻 Author

**Ravindra K**
Aspiring Machine Learning / AI Engineer

---

## ⭐ If you find this useful

Give the repository a ⭐ on GitHub and feel free to fork or contribute!
