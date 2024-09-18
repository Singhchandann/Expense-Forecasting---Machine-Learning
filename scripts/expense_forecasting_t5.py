import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('/kaggle/input/daily-transactions-dataset/Daily Household Transactions.csv')

# Select necessary columns
data = data[['Date', 'Category', 'Subcategory', 'Amount', 'Income/Expense']]

# Convert the `Date` column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['Date'])

# Sort the data by date
data = data.sort_values(by='Date')

# Create sequences of past expenses for forecasting
def create_sequences(data, n_past):
    sequences = []
    for i in range(n_past, len(data)):
        seq = data.iloc[i-n_past:i].to_dict('records')
        target = data.iloc[i]['Amount']
        sequences.append((seq, target))
    return sequences

n_past = 7  # Number of past days to use for predicting the next day's expense
sequences = create_sequences(data, n_past)

# Split data into training and validation sets
train_seqs, val_seqs = train_test_split(sequences, test_size=0.2, random_state=42)

from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def encode_sequences(sequences, tokenizer):
    inputs = []
    targets = []
    for seq, target in sequences:
        input_text = ' '.join([f"{item['Date'].strftime('%Y-%m-%d')} {item['Category']} {item['Subcategory']} {item['Amount']} {item['Income/Expense']}" for item in seq])
        target_text = str(target)
        inputs.append(tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True))
        targets.append(tokenizer.encode(target_text, return_tensors='pt', max_length=128, truncation=True))
    return inputs, targets

train_inputs, train_targets = encode_sequences(train_seqs, tokenizer)
val_inputs, val_targets = encode_sequences(val_seqs, tokenizer)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ExpenseForecastingDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx].squeeze(),
            'labels': self.targets[idx].squeeze()
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

train_dataset = ExpenseForecastingDataset(train_inputs, train_targets)
val_dataset = ExpenseForecastingDataset(val_inputs, val_targets)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

from transformers import T5ForConditionalGeneration, AdamW

model = T5ForConditionalGeneration.from_pretrained('t5-small')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(train_dataloader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW
import torch
# # Save the model
model.save_pretrained('expense_forecasting_t5')
tokenizer.save_pretrained('expense_forecasting_t5')

# Load the model
model = T5ForConditionalGeneration.from_pretrained('expense_forecasting_t5')
tokenizer = T5Tokenizer.from_pretrained('expense_forecasting_t5')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.to(device)

def predict_expenses(model, tokenizer, input_seq, num_days):
    predictions = []
    for _ in range(num_days):
        input_text = ' '.join([f"{item['Date'].strftime('%Y-%m-%d')} {item['Category']} {item['Subcategory']} {item['Amount']} {item['Income/Expense']}" for item in input_seq])
        input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
        output = model.generate(input_ids, max_length=50)
        predicted_amount = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Assuming the model returns a valid number
        try:
            predicted_amount = float(predicted_amount)
        except ValueError:
            predicted_amount = 0.0
        
        predictions.append(predicted_amount)
        
        # Update the input sequence for the next prediction
        next_date = input_seq[-1]['Date'] + pd.Timedelta(days=1)
        next_entry = {
            'Date': next_date,
            'Category': input_seq[-1]['Category'],  # Keeping the same category for simplicity
            'Subcategory': input_seq[-1]['Subcategory'],
            'Amount': predicted_amount,
            'Income/Expense': 'Expense'
        }
        input_seq.append(next_entry)
        input_seq.pop(0)  # Remove the oldest entry to keep the window size constant
    
    return predictions


import pandas as pd

# Example data
example_seq = data.iloc[-30:].to_dict('records')  # Using the last 30 records as context

# Predict next month's total expenses (30 days)
num_days_in_month = 30
predicted_monthly_expenses = predict_expenses(model, tokenizer, example_seq, num_days_in_month)
total_monthly_expense = sum(predicted_monthly_expenses)
print(f'Total predicted expense for the next month: {total_monthly_expense}')

# Predict next year's total expenses (365 days)
num_days_in_year = 365
predicted_yearly_expenses = predict_expenses(model, tokenizer, example_seq, num_days_in_year)
total_yearly_expense = sum(predicted_yearly_expenses)
print(f'Total predicted expense for the next year: {total_yearly_expense}')
