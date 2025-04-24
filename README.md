# 🌾 Crop Production Analysis and Prediction

This project focuses on the analysis and prediction of agricultural crop production using data preprocessing, exploratory data analysis (EDA), and machine learning models. It uses a real-world dataset from Indian agriculture.

## 📁 Dataset

- **File Used**: `apy.csv`
- **Source**: Government of India data.gov.in
- **Key Columns**: `Crop_Year`, `State_Name`, `District_Name`, `Season`, `Crop`, `Area`, `Production`

## 📊 Key Features

- Handles missing values and outliers (with special handling for Coconut)
- Calculates `Yield` (Production per Area)
- Performs EDA with insightful visualizations
- Implements regression and classification models

## 📈 Exploratory Data Analysis (EDA)

- 📉 **Production Distribution** (log scale)
- 🌍 **Area vs Production** scatter plot (log-log scale)
- 🥇 **Top Crops** by total production
- 📅 **Yearly Production Trends**
- 🏆 **Top Producing States**

All plots are saved locally (`plot1_*.png` to `plot7_*.png`).

## 🤖 Machine Learning Models

### 🔁 Regression: Random Forest
- **Target**: `Production`
- **Metric**: R², RMSE
- **Result**: Visual comparison of predicted vs actual values

### 🔢 Classification: K-Nearest Neighbors (KNN)
- **Target**: `Yield_Category` (Low, Medium, High)
- **Metric**: Accuracy, Confusion Matrix, Classification Report

## 🧠 Model Performance Summary

| Model           | Metric   | Value  |
|----------------|----------|-------- |
| Random Forest  | R²       | 98.     |
| Random Forest  | RMSE     | 17000   |
| KNN            | Accuracy | 85      |

*(Actual values printed in the console)*

## 🛠️ Tools & Libraries

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (RandomForestRegressor, KNeighborsClassifier)
- Jupyter / Spyder IDE

## 📌 Instructions to Run

1. Make sure `apy.csv` is present in your working directory.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
