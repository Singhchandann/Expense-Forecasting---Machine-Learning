import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import logging
import json
import warnings
import os
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df):
    logging.info("Performing data preprocessing...")
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('UTC')
        else:
            logging.info("Datetime column already timezone-aware.")
        current_datetime = datetime.now(timezone('UTC'))
        df['time_diff'] = (current_datetime - df['Date']).dt.days
        df['hours'] = df['Date'].dt.hour
        df['weekday'] = df['Date'].dt.weekday + 1
        df['year'] = df['Date'].dt.year
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['month'] = df['Date'].dt.month
        df.drop(['Date'], axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['Category'], drop_first=True)
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise e
    return df

def split_and_scale(df, target_column='Amount', test_size=0.2, random_state=42):
    logging.info("Splitting and scaling the data...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns.tolist()  # Save feature names for later use
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open('fitted_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)  # Save feature names to a JSON file
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    logging.info("Training and evaluating the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):", rmse)
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

if os.path.exists("./Daily Household Transactions.csv"):
    df = pd.read_csv("./Daily Household Transactions.csv")
    df = df.drop("Mode", axis=1)
    df = df.drop("Subcategory", axis=1)
    df = df.drop("Note", axis=1)
    df = df.drop("Currency", axis=1)
    df = df[df['Income/Expense'] == 'Expense']
    df = df.drop("Income/Expense", axis=1)
    df = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df)
    trained_model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
else:
    logging.error("Failed to find the file: Daily Household Transactions.csv")

# Code to load the model, scaler, and feature names for prediction
def load_model_scaler_and_features():
    with open('final_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('fitted_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model_scaler_and_features()

def preprocess_new_data(data_frame, feature_names):
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], errors='coerce')
    data_frame = data_frame.dropna(subset=['Date'])
    if data_frame['Date'].dt.tz is None:
        data_frame['Date'] = data_frame['Date'].dt.tz_localize('UTC')

    current_datetime = datetime.now(timezone('UTC'))
    data_frame['time_diff'] = (current_datetime - data_frame['Date']).dt.days
    data_frame['hours'] = data_frame['Date'].dt.hour
    data_frame['weekday'] = data_frame['Date'].dt.weekday + 1
    data_frame['year'] = data_frame['Date'].dt.year
    data_frame['day_of_year'] = data_frame['Date'].dt.dayofyear
    data_frame['month'] = data_frame['Date'].dt.month
    data_frame.drop(['Date'], axis=1, inplace=True)
    data_frame = pd.get_dummies(data_frame, columns=['Category'], drop_first=True)

    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in data_frame.columns:
            data_frame[feature] = 0

    # Reorder columns as per the training data
    data_frame = data_frame[feature_names]

    scaled_features = scaler.transform(data_frame)
    return scaled_features

def predict_next_year_by_category(model, scaler, feature_names):
    current_date = datetime.now()
    future_dates = [current_date + timedelta(days=30*i) for i in range(1, 13)]
    categories = [col for col in feature_names if 'Category_' in col]
    
    predictions = []

    for future_date in future_dates:
        for category in categories:
            # Initialize the DataFrame with zeros for all feature columns
            future_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

            # Set the future date related features
            future_df['time_diff'] = (future_date - current_date).days
            future_df['hours'] = future_date.hour
            future_df['weekday'] = future_date.weekday() + 1
            future_df['year'] = future_date.year
            future_df['day_of_year'] = future_date.timetuple().tm_yday
            future_df['month'] = future_date.month

            # Set the category feature
            future_df[category] = 1

            scaled_features = scaler.transform(future_df)
            prediction = model.predict(scaled_features)
            predictions.append({
                'month': future_date.strftime('%B'),
                'year': future_date.year,
                'category': category.split('Category_')[1],
                'predicted_amount': prediction[0]
            })

    return predictions

predictions = predict_next_year_by_category(model, scaler, feature_names)
predictions_df = pd.DataFrame(predictions)

# Aggregate predictions by month and category
monthly_category_totals = predictions_df.groupby(['year', 'month', 'category'])['predicted_amount'].sum().reset_index()
monthly_totals = predictions_df.groupby(['year', 'month'])['predicted_amount'].sum().reset_index()

print("Monthly Predictions by Category:")
print(monthly_category_totals)

print("Monthly Total Predictions:")
print(monthly_totals)
