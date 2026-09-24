import pickle
import numpy as np
import pandas as pd

# Load saved artifacts
with open('models/best_salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

def predict_salary(input_dict: dict) -> float:
    """
    Transforms raw input into dummy variables, aligns columns to match
    the training set, and returns estimated salary in thousands (USD).
    """
    input_df = pd.DataFrame([input_dict])
    encoded_df = pd.get_dummies(input_df)
    
    # Reindex against training features, filling missing dummy columns with 0
    final_features = encoded_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_features)
    return round(float(prediction[0]), 2)