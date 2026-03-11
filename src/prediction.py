#1. Load scaler.pkl and model.pkl from artifacts folder
#2. Create a class with a prediction function

import pickle
import numpy as np
import pandas as pd
import os

class Diabetes_Prediction:
    def __init__(self):
        # Get the directory of the current script and navigate to artifacts
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(script_dir, "..", "artifacts")

        with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(artifacts_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, BMI, Avg_Blood_Pressure, Total_Cholesterol,
                   LDL_Cholesterol, HDL_Cholesterol, TC_HDL_Ratio,
                   Log_Triglycerides, Blood_Sugar):

        Input = pd.DataFrame([[Age, BMI, Avg_Blood_Pressure, Total_Cholesterol,
                                LDL_Cholesterol, HDL_Cholesterol, TC_HDL_Ratio,
                                Log_Triglycerides, Blood_Sugar]],
                             columns=['Age', 'BMI', 'Avg_Blood_Pressure', 'Total_Cholesterol',
                                      'LDL_Cholesterol', 'HDL_Cholesterol', 'TC_HDL_Ratio',
                                      'Log_Triglycerides', 'Blood_Sugar'])
        Scaled_Input = self.scaler.transform(Input)
        result       = self.model.predict(Scaled_Input)
        return round(float(result[0]), 2)
