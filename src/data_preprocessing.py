#1. Load the Raw data
#2. Identifying X and y (input and output)
#3. Split X and y into training and test sets

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data():
    data = pd.read_csv("../data/raw/diabetes_data.csv")

    X = data[['Age', 'BMI', 'Avg_Blood_Pressure', 'Total_Cholesterol',
              'LDL_Cholesterol', 'HDL_Cholesterol', 'TC_HDL_Ratio',
              'Log_Triglycerides', 'Blood_Sugar']]
    y = data['Disease_Progression_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
