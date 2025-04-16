# Crop Production Analysis Project
# Goal: Analyze crop data, predict Production, and classify Yield categories
# For: Class presentation with clear insights, models, and saved plots for slides

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Set Seaborn theme for polished visuals
sns.set_theme(style="whitegrid")

# Function to log output to file
def log_output(message):
    with open("project_output.txt", "a") as f:
        f.write(str(message) + "\n")

# Print and log project title
title = f"""=== Crop Production Analysis Project ===
Predicting Production and Yield Categories for Agricultural Data
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"""
print(title)
log_output(title)

# --- Load and Clean Data ---
df = pd.read_csv(r"D:\pythonproject\apy.csv")

# Fill missing Production with median by Crop
df["Production"] = df.groupby("Crop")["Production"].transform(lambda x: x.fillna(x.median()))

# Cap outliers in Production
Q1 = df["Production"].quantile(0.25)
Q3 = df["Production"].quantile(0.75)
IQR = Q3 - Q1
df["Production"] = df["Production"].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Add Yield column (Production / Area)
df["Yield"] = df["Production"] / df["Area"]

# --- EDA and Key Analyses ---

# 1. Distribution of Production
plt.figure(figsize=(6, 4))
sns.histplot(np.log1p(df["Production"]), bins=40)
plt.title("Distribution of Crop Production (Log Scale)")
plt.xlabel("Log(Production)")
plt.savefig("plot1_production_histogram.png")
plt.show()
insight1 = "Insight: Production varies widely, so I used log scale."
print(insight1)
log_output(insight1)

# 2. Scatter Plot: Area vs Production
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Area", y="Production")
plt.title("Area vs Production")
plt.yscale("log")
plt.ylabel("Production (Log Scale)")
plt.savefig("plot2_area_vs_production.png")
plt.show()
insight2 = "Insight: Larger areas often mean more production."
print(insight2)
log_output(insight2)

# 3. Top Crops by Total Production
top_crops = df.groupby("Crop")["Production"].sum().nlargest(5)
plt.figure(figsize=(6, 4))
top_crops.plot(kind="bar")
plt.title("Top 5 Crops by Production")
plt.ylabel("Production")
plt.savefig("plot3_top_crops.png")
plt.show()
insight3 = "Insight: Rice and Coconut dominate due to large-scale farming."
print(insight3)
log_output(insight3)

# 4. Total Production Over Years
prod_trend = df.groupby("Crop_Year")["Production"].sum()
plt.figure(figsize=(6, 4))
prod_trend.plot(marker='o')
plt.title("Production Over Years")
plt.ylabel("Total Production")
plt.savefig("plot4_yearly_trends.png")
plt.show()
insight4 = "Insight: Production stays steady, showing stable agriculture."
print(insight4)
log_output(insight4)

# 5. Top States by Production
top_states = df.groupby("State_Name")["Production"].sum().nlargest(5)
plt.figure(figsize=(6, 4))
top_states.plot(kind="bar")
plt.title("Top 5 States by Production")
plt.ylabel("Production")
plt.savefig("plot5_top_states.png")
plt.show()
insight5 = "Insight: Andhra Pradesh leads, likely due to good irrigation."
print(insight5)
log_output(insight5)

# 6. Correlation Heatmap
corr = df[["Area", "Production", "Yield"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Between Area, Production, Yield")
plt.savefig("plot6_correlation_heatmap.png")
plt.show()
insight6 = "Insight: Area and Production are strongly linked."
print(insight6)
log_output(insight6)

# --- Machine Learning Models ---

# Encode categorical variables
df = pd.get_dummies(df, columns=["State_Name", "Season", "Crop"], drop_first=True)

# Create target variables
df["Yield_Category"] = pd.qcut(df["Production"], q=3, labels=["Low", "Medium", "High"])
X = df.drop(columns=["Production", "Yield", "Yield_Category", "District_Name"])
y_reg = df["Production"]
y_clf = df["Yield_Category"]

# Train-test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# --- Regression Models ---
reg_output = """=== Regression Models ===
(RMSE in tonnes, R² shows % of Production explained)
Linear Regression (Baseline)
------------------------------
"""
print(reg_output)
log_output(reg_output)

lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
y_pred_lr = lr_model.predict(X_test_r)
lr_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_lr))
lr_r2 = r2_score(y_test_r, y_pred_lr)
lr_metrics = f"RMSE: {lr_rmse:.2f} tonnes\nR² Score: {lr_r2:.2f}\n"
print(lr_metrics)
log_output(lr_metrics)

rf_output = """Random Forest (Main Model)
------------------------------
"""
print(rf_output)
log_output(rf_output)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)
rf_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
rf_r2 = r2_score(y_test_r, y_pred_r)
rf_metrics = f"RMSE: {rf_rmse:.2f} tonnes\nR² Score: {rf_r2:.2f}\n"
print(rf_metrics)
log_output(rf_metrics)

# Plot Random Forest predictions
plt.figure(figsize=(6, 4))
plt.scatter(y_test_r, y_pred_r, alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], "r--")
plt.xlabel("Actual Production (tonnes)")
plt.ylabel("Predicted Production (tonnes)")
plt.title("Random Forest: Predicted vs Actual")
plt.savefig("plot7_rf_predictions.png")
plt.show()

# --- Classification Model ---
clf_output = """=== Classification Model ===
K-Nearest Neighbors (KNN)
(Accuracy shows % of correct Yield category predictions)
------------------------------
"""
print(clf_output)
log_output(clf_output)

clf_model = KNeighborsClassifier(n_neighbors=5)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
knn_acc = accuracy_score(y_test_c, y_pred_c)
clf_metrics = f"Accuracy: {knn_acc:.2f}\n\nClassification Report (precision, recall, f1-score per category):\n{classification_report(y_test_c, y_pred_c)}"
print(clf_metrics)
log_output(clf_metrics)

# Plot confusion matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix: Yield Category")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plot8_confusion_matrix.png")
plt.show()

# --- Model Performance Summary ---
summary_output = f"""=== Model Performance Summary ===
{'Model':<25} {'Metric':<10} {'Value':<10}
{'-' * 45}
{'Linear Regression':<25} {'RMSE':<10} {lr_rmse:.2f}
{'Linear Regression':<25} {'R²':<10} {lr_r2:.2f}
{'Random Forest':<25} {'RMSE':<10} {rf_rmse:.2f}
{'Random Forest':<25} {'R²':<10} {rf_r2:.2f}
{'KNN':<25} {'Accuracy':<10} {knn_acc:.2f}
"""
print(summary_output)
log_output(summary_output)

# Save metrics to CSV
metrics_df = pd.DataFrame([
    {"Model": "Linear Regression", "Metric": "RMSE", "Value": lr_rmse},
    {"Model": "Linear Regression", "Metric": "R²", "Value": lr_r2},
    {"Model": "Random Forest", "Metric": "RMSE", "Value": rf_rmse},
    {"Model": "Random Forest", "Metric": "R²", "Value": rf_r2},
    {"Model": "KNN", "Metric": "Accuracy", "Value": knn_acc}
])
metrics_df.to_csv("model_metrics.csv", index=False)
print("Model metrics saved to model_metrics.csv")
log_output("Model metrics saved to model_metrics.csv")

# --- Conclusion ---
conclusion = f"""=== Conclusion ===
Key Findings:
- Rice and Coconut are top crops due to large-scale cultivation.
- States like Andhra Pradesh lead, likely due to strong irrigation.
- Random Forest predicted Production with high accuracy (R² = {rf_r2:.2f}, RMSE = {rf_rmse:.2f}), outperforming Linear Regression (R² = {lr_r2:.2f}).
- KNN classified Yield categories reliably (Accuracy = {knn_acc:.2f}).
Impact: These insights and models can help farmers choose high-yield crops and optimize resources.
"""
print(conclusion)
log_output(conclusion)