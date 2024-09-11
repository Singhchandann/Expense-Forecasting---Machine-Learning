import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def train_custom_model(input_file, model_type, **kwargs):
    logging.info(f"Reading processed data from {input_file}")
    
    df = pd.read_csv(input_file)

    # Features and target
    X = df.drop('Amount', axis=1)
    y = df['Amount']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose the model based on input type
    if model_type == 'rf':
        logging.info("Training Random Forest model with custom hyperparameters.")
        model = RandomForestRegressor(**kwargs)
    elif model_type == 'xgboost':
        logging.info("Training XGBoost model with custom hyperparameters.")
        model = xgb.XGBRegressor(**kwargs)
    else:
        raise ValueError("Unsupported model type. Choose 'rf' or 'xgboost'.")

    model.fit(X_train_scaled, y_train)

    model_filename = f"models/final_model_{model_type}.pkl"
    # Save the model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    # Save the scaler
    with open('models/fitted_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logging.info(f"{model_type.upper()} model trained and saved as {model_filename}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a custom model (Random Forest or XGBoost) with hyperparameters")
    parser.add_argument('--input_file', required=True, help="Path to the preprocessed data CSV file")
    parser.add_argument('--model_type', choices=['rf', 'xgboost'], required=True, help="Model type (Random Forest or XGBoost)")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of estimators (trees)")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate (for XGBoost)")
    args = parser.parse_args()

    kwargs = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate if args.model_type == 'xgboost' else None
    }
    
    train_custom_model(args.input_file, args.model_type, **kwargs)
