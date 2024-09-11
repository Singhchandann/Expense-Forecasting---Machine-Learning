# Expense Forecasting Project

This project aims to predict future expenses based on past financial data using machine learning models, including **Random Forest**, **XGBoost**, and **Custom Models** (Random Forest or XGBoost with hyperparameter tuning). It processes transaction data, fits models on it, and predicts future expenses.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [1. Preprocessing the Data](#preprocessing-the-data)
  - [2. Training the Model](#training-the-model)
    - [Training Random Forest](#training-random-forest)
    - [Training XGBoost](#training-xgboost)
    - [Training Custom Model](#training-custom-model)
  - [3. Predicting Future Expenses](#predicting-future-expenses)
    - [Prediction using Random Forest](#prediction-using-random-forest)
    - [Prediction using XGBoost](#prediction-using-xgboost)
    - [Prediction using Custom Model](#prediction-using-custom-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

expense_forecasting_project/
│  
├── README.md  
├── data/  
│   └── sample_data.csv  
├── models/  
│   ├── final_model_rf.pkl  
│   ├── final_model_xgboost.pkl  
│   └── fitted_scaler.pkl  
├── scripts/  
│   ├── preprocess.py  
│   ├── train_model_rf.py  
│   ├── train_model_xgboost.py  
│   ├── train_model_custom.py  
│   ├── predict_rf.py  
│   ├── predict_xgboost.py  
│   ├── predict_custom.py  
│   └── load_model.py  
├── requirements.txt  
├── LICENSE 
└── .gitignore

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Singhchandann/Expense-Forecasting--Machine-Learning.git
    cd expense_forecasting_project
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data

The data used is daily financial transactions, containing columns like `Date`, `Category`, `Subcategory`, `Amount`, and `Income/Expense`. A sample dataset (`sample_data.csv`) is included in the `data/` folder.

You can replace the dataset with your own CSV file.

## Usage

### 1. Preprocessing the Data

Run the preprocessing script:

```bash
python scripts/preprocess.py --input_file data/sample_data.csv --output_file data/processed_data.csv
```
This will clean and transform the data, and the preprocessed data will be saved as processed_data.csv.

### 2. Training the Model

To train the model on preprocessed data:

# Training Random Forest

```bash
python scripts/train_model_rf.py --input_file data/processed_data.csv
```
This will train the Random Forest model, and the trained model and scaler will be saved in the models/ directory.

# Training XGBoost

```bash
python scripts/train_model_xgboost.py --input_file data/processed_data.csv
```
This trains an XGBoost model and saves the model to the models/ directory.

# Training Custom Model

```bash
python scripts/train_model_custom.py --input_file data/processed_data.csv --model_type rf/xgboost --param1 value1 ...
```
This script allows you to specify either a Random Forest (rf) or XGBoost (xgboost) model with custom hyperparameters.

### 3. Predicting Future Expenses

# Prediction using Random Forest
```bash
python scripts/predict_rf.py --input_file data/processed_data.csv
```

# Prediction using XGBoost
```bash
python scripts/predict_xgboost.py --input_file data/processed_data.csv
```

# Prediction using Custom Model
```bash
python scripts/predict_custom.py --input_file data/processed_data.csv --model_type rf/xgboost
```

## Results
The predictions will be saved as CSV files in the results/ directory.

After running the model, you will receive predictions for each category in the form of a CSV file. Here’s a sample of what the predictions might look like:

Month	Category	Predicted Amount  
Jan	     Groceries  	$400  
Feb	   Entertainment	$150  
...	...	...

## Contributing
We welcome contributions to this project. Please feel free to submit pull requests for new features or improvements.
