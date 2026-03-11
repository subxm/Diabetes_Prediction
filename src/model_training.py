#1. Load the processed data from processed folder
#2. Create model and train on data
#3. Evaluate the model
#4. Save the model into artifacts folder

import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test  = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test  = pd.read_csv("../data/processed/y_test.csv")

print("Training data shape :", X_train.shape)

# Train Ridge Regression model (alpha=1.0 handles multicollinearity in medical data)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation:")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.2f}")
print(f"  RMSE     : {rmse:.2f}")

# Save model
with open("../artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSuccessfully saved model.pkl!")
