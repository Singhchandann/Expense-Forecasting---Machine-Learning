import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def train_random_forest(input_file):
    df = pd.read_csv(input_file)
    X = df.drop('Amount', axis=1)
    y = df['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    with open('models/final_model_rf.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/fitted_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logging.info("Random Forest model trained and saved.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train the Random Forest model")
    parser.add_argument('--input_file', required=True, help="Path to the processed data CSV file")
    args = parser.parse_args()

    train_random_forest(args.input_file)
