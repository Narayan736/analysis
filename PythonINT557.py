import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# --- Load and Clean Data ---
df = pd.read_csv(r"D:\pythonproject\apy.csv")

# Fill missing Production with median by Crop
df["Production"] = df.groupby("Crop")["Production"].transform(lambda x: x.fillna(x.median()))

# Cap outliers in Production, but skip Coconut
def clip_outliers(group, crop_name):
    if crop_name == "Coconut":
        return group
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    return group.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

df["Production"] = df.groupby("Crop")["Production"].transform(lambda x: clip_outliers(x, x.name))

# Add Yield column
df["Yield"] = df["Production"] / df["Area"]

# --- EDA and Key Analyses ---

# 1. Distribution of Production
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df["Production"]), bins=40, color="dodgerblue")
plt.title("Distribution of Crop Production (Log Scale)", fontsize=14)
plt.xlabel("Log(Production)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
plt.savefig("plot1_production_histogram.png")
plt.show()
print("Insight: Production varies widely, so I used log scale.")

# 2. Scatter Plot: Area vs Production
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area", y="Production", alpha=0.5, color="dodgerblue", s=50)
plt.xscale("log")
plt.yscale("log")
plt.title("Area vs Production (Log-Log Scale)", fontsize=14)
plt.xlabel("Area (Hectares, Log Scale)", fontsize=12)
plt.ylabel("Production (Tonnes, Log Scale)", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
sns.regplot(data=df, x="Area", y="Production", scatter=False, color="red", logx=True)
plt.text(0.5, -0.25, "Larger areas often mean more production.", ha="center", fontsize=10, transform=plt.gca().transAxes)
plt.savefig("plot2_area_vs_production.png", bbox_inches="tight")
plt.show()
print("Insight: Larger areas often mean more production.")

# 3. Top Crops by Total Production (Updated Insight and Caption)
top_crops = df.groupby("Crop")["Production"].sum().nlargest(5)
plt.figure(figsize=(8, 5))
top_crops.plot(kind="bar", color="dodgerblue")
plt.title("Top 5 Crops by Production", fontsize=14)
plt.ylabel("Production (Tonnes)", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
plt.text(0.5, -0.25, "Coconut, Sugarcane, and Rice lead!", ha="center", fontsize=10, transform=plt.gca().transAxes)
plt.savefig("plot3_top_crops.png", bbox_inches="tight")
plt.show()
print("Insight: Coconut, Sugarcane, and Rice dominate due to large-scale farming.")

# 4. Total Production Over Years
prod_trend = df.groupby("Crop_Year")["Production"].sum()
plt.figure(figsize=(8, 5))
prod_trend.plot(marker='o', color="dodgerblue")
plt.title("Production Over Years", fontsize=14)
plt.ylabel("Total Production", fontsize=12)
plt.xlabel("Crop_Year", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
plt.savefig("plot4_yearly_trends.png")
plt.show()
print("Insight: Production fluctuates over the years, with a peak around 2007–2010.")

# 5. Top States by Production
top_states = df.groupby("State_Name")["Production"].sum().nlargest(5)
plt.figure(figsize=(8, 5))
top_states.plot(kind="bar", color="dodgerblue")
plt.title("Top 5 States by Production", fontsize=14)
plt.ylabel("Production", fontsize=12)
plt.xlabel("State_Name", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
plt.savefig("plot5_top_states.png")
plt.show()
print("Insight: Kerala and Andhra Pradesh lead thanks to strong agricultural practices.")

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

# --- Regression Model (Random Forest) ---
print("\n=== Regression Model (Random Forest) ===")
reg_model = RandomForestRegressor(n_estimators=20, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)
rf_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
rf_r2 = r2_score(y_test_r, y_pred_r)
print(f"RMSE: {rf_rmse:.2f} tonnes\nR² Score: {rf_r2:.2f}")

# Plot Random Forest predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_r, alpha=0.5, color="dodgerblue", s=50)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], "r--")
plt.xlabel("Actual Production (tonnes)", fontsize=12)
plt.ylabel("Predicted Production (tonnes)", fontsize=12)
plt.title("Random Forest: Predicted vs Actual", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.text(0.5, -0.25, f"R² = {rf_r2:.2f}, RMSE = {rf_rmse:.2f} tonnes", ha="center", fontsize=10, transform=plt.gca().transAxes)
plt.savefig("plot6_rf_predictions.png", bbox_inches="tight")
plt.show()

# --- Classification Model (KNN) ---
print("\n=== Classification Model (KNN) ===")
clf_model = KNeighborsClassifier(n_neighbors=5)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
knn_acc = accuracy_score(y_test_c, y_pred_c)
print(f"Accuracy: {knn_acc:.2f}")
print(f"Classification Report:\n{classification_report(y_test_c, y_pred_c)}")

# Plot confusion matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix: Yield Category", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tick_params(axis='both', labelsize=10)
plt.text(0.5, -0.25, f"Accuracy = {knn_acc:.2f}", ha="center", fontsize=10, transform=plt.gca().transAxes)
plt.savefig("plot7_confusion_matrix.png", bbox_inches="tight")
plt.show()

# --- Summary (Formatted Table) ---
print("\n=== Model Performance Summary ===")
print(f"{'Model':<25} {'Metric':<10} {'Value':<10}")
print("-" * 45)
print(f"{'Random Forest':<25} {'R²':<10} {rf_r2:.2f}")
print(f"{'Random Forest':<25} {'RMSE':<10} {rf_rmse:.2f}")
print(f"{'KNN':<25} {'Accuracy':<10} {knn_acc:.2f}")
