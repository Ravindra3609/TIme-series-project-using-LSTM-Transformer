import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
url = "https://data.cityofchicago.org/api/views/6iiy-9s97/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url, parse_dates=["service_date"])
print(df.head())

df_filtered = df[df['service_date'] <= '2019-12-31']

print("Filtered DataFrame head:")
print(df_filtered.head())

print("\nShape of the filtered DataFrame:", df_filtered.shape)
df = df_filtered

df.sort_values("service_date", inplace=True)
ts = df.set_index("service_date")["total_rides"].fillna(0)

plt.plot(ts)
plt.title("CTA Daily Total Rides")
plt.show()

n = len(ts)
train = ts[:int(0.8*n)]
test = ts[int(0.8*n):]

train_vals = train.values.astype(float)
test_vals = test.values.astype(float)

def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X_train, y_train = create_sequences(train_vals, SEQ_LEN)
X_test, y_test = create_sequences(test_vals, SEQ_LEN)

# Convert our formatted data into PyTorch tensors
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float().unsqueeze(-1)
X_test = torch.tensor(X_test).float().unsqueeze(-1)
y_test = torch.tensor(y_test).float().unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

lstm_model = LSTMModel()

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])

transformer_model = SimpleTransformer()

def train(model, X, y, epochs=10):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return model

lstm_model = train(lstm_model, X_train, y_train)
transformer_model = train(transformer_model, X_train, y_train)

lstm_model.eval()
transformer_model.eval()

pred_lstm = lstm_model(X_test).detach().numpy().flatten()
pred_trans = transformer_model(X_test).detach().numpy().flatten()
true_vals = y_test.numpy().flatten()

rmse_lstm = np.sqrt(mean_squared_error(true_vals, pred_lstm))
mae_lstm  = mean_absolute_error(true_vals, pred_lstm)

rmse_trans = np.sqrt(mean_squared_error(true_vals, pred_trans))
mae_trans  = mean_absolute_error(true_vals, pred_trans)

print(f"LSTM RMSE={rmse_lstm:.1f}, MAE={mae_lstm:.1f}")
print(f"Trans RMSE={rmse_trans:.1f}, MAE={mae_trans:.1f}")
