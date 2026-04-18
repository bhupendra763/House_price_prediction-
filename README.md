# 🏠 House Price Prediction — Deep Learning (PyTorch ANN)

A deep learning project that predicts house sale prices using an Artificial Neural Network (ANN) built with PyTorch.

---

## 📌 Project Overview

This project uses a real estate dataset to train a neural network that predicts the **SalePrice** of a house based on features like lot area, year built, building type, and more.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── House_price.csv          # Dataset
├── house_price_complete.py  # Full training pipeline
├── best_model.pt            # Saved best model weights
├── scaler.pkl               # Saved StandardScaler
└── README.md                # Project documentation
```

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `MSSubClass` | Type of dwelling |
| `MSZoning` | General zoning classification |
| `LotArea` | Lot size in square feet |
| `LotConfig` | Lot configuration |
| `BldgType` | Type of dwelling |
| `OverallCond` | Overall condition rating |
| `YearBuilt` | Original construction year |
| `YearRemodAdd` | Remodel year |
| `Exterior1st` | Exterior covering on house |
| `BsmtFinSF2` | Basement finished area |
| `TotalBsmtSF` | Total basement area |
| `SalePrice` | ⭐ Target variable |

- Total rows: **2,919** (1,460 labeled)
- After cleaning: **1,454 rows used for training**

---

## 🧠 Model Architecture

```
Input Layer  →  128 neurons (BatchNorm + ReLU + Dropout 0.3)
             →  64  neurons (BatchNorm + ReLU + Dropout 0.3)
             →  32  neurons (ReLU)
Output Layer →  1   neuron  (SalePrice prediction)
```

| Parameter | Value |
|---|---|
| Loss Function | MSELoss |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Epochs | 200 |
| Batch Size | 32 |
| Train/Test Split | 80% / 20% |

---

## ✅ Results

| Metric | Value |
|---|---|
| Training MSE | 785,167,936 |
| Testing MSE | 2,166,434,816 |
| **R² Score** | **0.7175** |

> The model explains ~72% of the variance in house prices. MSE is large because prices are in raw dollar values (100k–700k range).

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn joblib
```

### 3. Run the training script

```bash
python house_price_complete.py
```

### 4. Output

- `best_model.pt` — saved model weights
- `scaler.pkl` — saved scaler for inference on new data
- Loss curve plot shown after training

---

## 🔮 Making Predictions on New Data

```python
import torch
import joblib
import pandas as pd

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = ANN(input_dim=34)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Prepare new data (must have same columns as training data)
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction = model(X_new_tensor)
    print("Predicted Price: $", round(prediction.item(), 2))
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

---

## 📈 Future Improvements
- Try other architectures (deeper networks, residual connections)
- Add cross-validation
- Deploy as a web app using Flask or Streamlit

---

## 👤 Author

**Your Name**
- GitHub: [@bhupendra763](https://github.com/bhupendra763)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
