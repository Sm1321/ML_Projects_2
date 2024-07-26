import streamlit as st
import pandas as pd
import pickle
from joblib import dump, load


import numpy as np
from sklearn.preprocessing import LabelEncoder


# Path to your saved model
file_path = r"C:/All_PRojects_/1.Machine Learning Projects/5.Big Mart Sales Prediction using ML/Best_model_GB.joblib"

# Load the model
try:
    best_model = load(file_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")


# Initialize the LabelEncoder
encoder = LabelEncoder()

# Function to preprocess the data
def preprocess_data(df):
    # Handle missing values
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

    # Encode categorical variables uisng the LAber Encoder
    df['Item_Fat_Content'] = encoder.fit_transform(df['Item_Fat_Content'])
    df['Item_Identifier'] = encoder.fit_transform(df['Item_Identifier'])
    df['Item_Type'] = encoder.fit_transform(df['Item_Type'])
    df['Outlet_Identifier'] = encoder.fit_transform(df['Outlet_Identifier'])
    df['Outlet_Size'] = encoder.fit_transform(df['Outlet_Size'])
    df['Outlet_Location_Type'] = encoder.fit_transform(df['Outlet_Location_Type'])
    df['Outlet_Type'] = encoder.fit_transform(df['Outlet_Type'])

    return df

# Streamlit app
def main():
    st.title('Big Mart Sales Prediction')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your test data (CSV file)", type="csv")
    
    if uploaded_file is not None:
        # Read the file into a DataFrames
        data = pd.read_csv(uploaded_file)
        
        # Display the data
        st.write("Uploaded Data", data.head())
        
        # Preprocess the data
        preprocessed_data = preprocess_data(data)
        
        # Drop the 'Item_Outlet_Sales' column if it exists
        if 'Item_Outlet_Sales' in preprocessed_data.columns:
            preprocessed_data = preprocessed_data.drop(columns='Item_Outlet_Sales')
        
        # Make predictions for the given data
        predictions = best_model.predict(preprocessed_data)
        
        # Display predictions 
        st.write("Predictions", predictions)

     

if __name__ == "__main__":
    main()


