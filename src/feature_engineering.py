#1. Load the training and testing data
#2. Scale the training data using StandardScaler
#3. Save scaled data into processed folder
#4. Save the scaler into artifacts folder

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_split_data

X_train, X_test, y_train, y_test = load_and_split_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform() scales data to mean=0, std=1
X_test_scaled  = scaler.transform(X_test)        # only transform (no fit) on test data

# Save scaled data into processed folder
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("../data/processed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled,  columns=X_test.columns).to_csv("../data/processed/X_test.csv",  index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",   index=False)

# Save the scaler
with open("../artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Successfully saved scaler.pkl and processed data!")
