import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create directories
folders = ["models", "visualization", "predictions", "logs"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Load dataset
file_path = "data/Housing.csv"
df = pd.read_csv(file_path)

# Convert categorical Yes/No to binary (1/0)
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in categorical_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Encode furnishingstatus as numerical
df["furnishingstatus"] = df["furnishingstatus"].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# Split data into features (X) and target variable (y)
X = df.drop(columns=["price"])
y = df["price"]

# Normalize numerical features
scaler = MinMaxScaler()
X[["area", "bedrooms", "bathrooms", "stories", "parking"]] = scaler.fit_transform(X[["area", "bedrooms", "bathrooms", "stories", "parking"]])
joblib.dump(scaler, "models/scaler.pkl")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel='linear')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}_model.pkl")

# Save model evaluation logs
with open("logs/model_performance.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name}: MSE={metrics['MSE']}, R2={metrics['R2']}\n")

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=models["Linear Regression"].predict(X_test))
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Linear Regression)")
plt.savefig("visualization/actual_vs_predicted.png")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("visualization/correlation_matrix.png")
plt.show()

# Feature importance for Random Forest
rf_model = models["Random Forest Regressor"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', title="Feature Importance (Random Forest)", figsize=(10, 5))
plt.savefig("visualization/feature_importance.png")
plt.show()

# Predict sample houses
sample_data = pd.DataFrame({
    "area": [1500, 2000],
    "bedrooms": [3, 4],
    "bathrooms": [2, 3],
    "stories": [2, 2],
    "mainroad": [1, 0],
    "guestroom": [1, 1],
    "basement": [0, 1],
    "hotwaterheating": [1, 0],
    "airconditioning": [1, 1],
    "parking": [1, 2],
    "prefarea": [1, 0],
    "furnishingstatus": [1, 2]
})

# Normalize sample data
sample_data[["area", "bedrooms", "bathrooms", "stories", "parking"]] = scaler.transform(sample_data[["area", "bedrooms", "bathrooms", "stories", "parking"]])

# Predict prices using best model (Random Forest assumed best)
predicted_prices = rf_model.predict(sample_data)
pd.DataFrame({"Sample": [1, 2], "Predicted Price": predicted_prices}).to_csv("predictions/sample_predictions.csv", index=False)
print("Predicted House Prices saved to predictions/sample_predictions.csv")
