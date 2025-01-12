import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load from yahoo
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for {ticker}. Check the ticker or date range.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        exit()

company = 'MSFT'
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2023, 1, 1)  # Reduced end date for more realistic test data
data = fetch_data(company, start, end)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 1
x, y = [], []

for i in range(prediction_days, len(scaled_data)):
    x.append(scaled_data[i - prediction_days:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = torch.from_numpy(x).float().unsqueeze(-1).to(device)
y = torch.from_numpy(y).float().to(device)


# Split data into training and validation sets (80/20 split)
train_size = int(0.8 * len(x))
x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=4, dropout_rate=0.5):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Learning rate scheduler

epochs = 50
best_val_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_function(predictions, batch_y.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y.unsqueeze(-1))
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    scheduler.step(avg_val_loss)  # Update learning rate based on validation loss

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())

# Load best model state
model.load_state_dict(best_model_state)

test_data = fetch_data(company, start=end, end=dt.datetime.now())
total_data = pd.concat((data['Close'], test_data['Close']), axis=0)


inputs = total_data[len(total_data) - len(test_data) - prediction_days:].values # Corrected
inputs = scaler.transform(inputs.reshape(-1, 1))

x_test = []
for i in range(prediction_days, len(inputs)):
    x_test.append(inputs[i - prediction_days:i, 0])

x_test = torch.from_numpy(np.array(x_test)).float().unsqueeze(-1)

model.eval()
with torch.no_grad():
    predictions = model(x_test).squeeze()
    predictions = scaler.inverse_transform(predictions.numpy().reshape(-1, 1))

# Calculate Percent Accuracy of model vs actual prices
actual_prices = test_data['Close'].values
predicted_prices = predictions.flatten()


if len(actual_prices) > len(predicted_prices):
    actual_prices = actual_prices[:len(predicted_prices)]

#prediction accuracy
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
accuracy = 100 - mape

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Prediction Accuracy: {accuracy:.2f}%")

#Investment Returns vs Actual Returns
def calculate_returns(investment, actual_prices, predicted_prices):

    num_shares_actual = investment / actual_prices[0]
    num_shares_predicted = investment / predicted_prices[0]

    final_actual = num_shares_actual * actual_prices[-1]
    final_predicted = num_shares_predicted * predicted_prices[-1]

    actual_profit = final_actual - investment
    predicted_profit = final_predicted - investment

    return actual_profit.item(), predicted_profit.item(), final_actual.item(), final_predicted.item()

investment_amount = float(input("Enter the amount you want to invest ($): "))

actual_profit, predicted_profit, final_actual, final_predicted = calculate_returns(
    investment_amount, actual_prices, predicted_prices
)

print(f"\nInvestment Results with ${investment_amount:.2f} investment:")
print(f"Actual Profit: ${actual_profit:.2f} | Final Value: ${final_actual:.2f}")
print(f"Predicted Profit: ${predicted_profit:.2f} | Final Value: ${final_predicted:.2f}")

def multi_day_predictions(model, inputs, prediction_days, scaler, days_to_predict):
    current_sequence = inputs[-prediction_days:].reshape(1, prediction_days, 1)  #Correct Reshape input to 3D for LSTM, ensure shape is matching.
    predictions = []

    model.eval()
    with torch.no_grad():
        for _ in range(days_to_predict):
            current_sequence_tensor = torch.tensor(current_sequence, dtype=torch.float32)
            next_price_scaled = model(current_sequence_tensor).item()
            next_price = scaler.inverse_transform([[next_price_scaled]])[0][0]
            
            predictions.append(next_price)

             # Update current sequence with new predicted price.
            next_input_scaled = scaler.transform(np.array([[next_price]])).reshape(1,1,1)
            current_sequence = np.concatenate((current_sequence[:, 1:, :], next_input_scaled), axis=1)
           
            
    return predictions

days_to_predict = int(input("Enter the number of days to predict: "))
future_prices = multi_day_predictions(model, inputs, prediction_days, scaler, days_to_predict)

# Print each day prediction
print("\nFuture Predictions:")
for i, price in enumerate(future_prices, 1):
    print(f"Day {i}: ${price:.2f}")

#graph
last_date = test_data.index[-1]
future_dates = [last_date + dt.timedelta(days=i) for i in range(1, days_to_predict + 1)]

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

plt.plot(
    test_data.index,
    test_data['Close'].values,
    color='limegreen',
    linewidth=2,
    label=f"Actual {company} Price"
)
plt.plot(
    test_data.index,
    predicted_prices,
    color='royalblue',
    linewidth=2,
    label=f"Predicted {company} Price"
)

plt.plot(
    future_dates,
    future_prices,
    color='royalblue',
    linestyle='dashed',
    linewidth=2
)
plt.grid(False)

plt.title(f"{company} Share Price Prediction", color='white', fontsize=16, pad=20)
plt.xlabel('', color='white', fontsize=0)  #
plt.ylabel('', color='white', fontsize=0)

plt.xticks(fontsize=10, color='gray')
plt.yticks(fontsize=10, color='gray')


for spine in plt.gca().spines.values():
    spine.set_visible(False)
for line in plt.gca().get_lines():
    line.set_alpha(0.85)
plt.tight_layout()
plt.legend()

plt.show()
