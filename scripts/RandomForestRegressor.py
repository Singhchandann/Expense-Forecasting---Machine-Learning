import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data from CSV file
file_path = r"./Daily Household Transactions.csv" # Replace with your actual file path
df = pd.read_csv(file_path)

# Ensure Date is in datetime format and handle various formats
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['Date'])

# Filter only expenses
df = df[df['Income/Expense'] == 'Expense']

# Extract month and year
df['YearMonth'] = df['Date'].dt.to_period('M')

# Aggregate the data by month
monthly_expense = df.groupby(['YearMonth', 'Category']).agg({'Amount': 'sum'}).reset_index()

# Pivot the data to have categories and subcategories as columns
pivot_df = monthly_expense.pivot(index='YearMonth', columns=['Category'], values='Amount').fillna(0)

# Ensure the index is a datetime object
pivot_df.index = pivot_df.index.to_timestamp()

# Aggregate to get total monthly spend
total_monthly_expense = pivot_df.sum(axis=1).to_frame(name='Total')

# Create features and target variable for total spend
def create_features_targets(df, n_months=1):
    X, y = [], []
    for i in range(len(df) - n_months):
        X.append(df.iloc[i:i + n_months].values.flatten())
        y.append(df.iloc[i + n_months].values)
    return np.array(X), np.array(y)

# Function to predict future expenses
def predict_future_expenses(df, n_months_to_predict=1, n_past_months=3):
    X, y = create_features_targets(df, n_past_months)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict future expenses
    last_n_months = df.iloc[-n_past_months:].values.flatten().reshape(1, -1)
    future_predictions = []
    
    for _ in range(n_months_to_predict):
        next_month_pred = model.predict(last_n_months)
        future_predictions.append(next_month_pred.flatten())
        last_n_months = np.roll(last_n_months, -df.shape[1])
        last_n_months[0, -df.shape[1]:] = next_month_pred
        
    return np.array(future_predictions)

# Predict future total expenses
n_months_to_predict = 3  # Predict the next 1 month
total_predicted_expenses = predict_future_expenses(total_monthly_expense, n_months_to_predict)

# Print predicted total expense for next month
print(f'Predicted total expense for next month: {total_predicted_expenses[0][0]}')

# Now predict the expenses for each category based on the predicted total
# Note: We will use the same function but on the pivot_df
category_predicted_expenses = predict_future_expenses(pivot_df, n_months_to_predict)

# Print predicted expenses for each category for the next month
print(f'Predicted expenses for each category for next month:')
category_expense_df = pd.DataFrame(category_predicted_expenses[0], index=pivot_df.columns).T
category_expense_df
