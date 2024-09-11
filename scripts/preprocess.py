import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

def preprocess_data(input_file, output_file):
    logging.info(f"Reading data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert 'Date' to datetime format
    logging.info("Standardizing date format.")
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Encoding categorical variables like Mode, Category, Subcategory
    label_encoders = {}
    for column in ['Mode', 'Category', 'Subcategory']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Add new columns such as day, month, and year based on the Date
    logging.info("Adding 'Day', 'Month', and 'Year' columns.")
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Drop 'Date' column after extracting day, month, and year
    df.drop(columns=['Date'], inplace=True)

    # Convert 'Income/Expense' to binary
    df['Income/Expense'] = df['Income/Expense'].apply(lambda x: 1 if x == 'Income' else 0)

    # Saving preprocessed data
    logging.info(f"Saving processed data to {output_file}")
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess daily transaction data")
    parser.add_argument('--input_file', required=True, help="Path to the CSV file containing raw transaction data")
    parser.add_argument('--output_file', required=True, help="Path to save the preprocessed CSV file")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file)
