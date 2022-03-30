# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# %%
x = np.linspace(0, 50, 501)
y = np.sin(x)

df = pd.DataFrame(data=y, index=x, columns=["Sine"])
test_percent = 0.1
test_ind = int(len(df) * (1 - test_percent))

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

length = 49
batch_size = 1

generator = TimeseriesGenerator(
    data=train_scaled, targets=train_scaled, length=length, batch_size=batch_size
)
# %%
n_features = 1
model = Sequential()

model.add(SimpleRNN(units=49, input_shape=(length, n_features)))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mse")

model.summary()
# %%
model.fit(generator, epochs=5)

# %%
losses = pd.DataFrame(model.history.history)
losses.plot()

# %%
# predict

test_pred = []
first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_pred.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_pred = scaler.inverse_transform(test_pred)
test["pred"] = true_pred
test.plot(figsize=(12, 10))
# %%
# Improvements

early_stop = EarlyStopping(monitor="val_loss", patience=2)

length = 49

generator = TimeseriesGenerator(
    data=train_scaled, targets=train_scaled, length=length, batch_size=batch_size
)

validation_gen = TimeseriesGenerator(
    data=test_scaled, targets=test_scaled, length=length, batch_size=batch_size
)
# %%
model = Sequential()

model.add(LSTM(units=49, input_shape=(length, n_features)))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mse")

model.summary()
# %%
model.fit(generator, epochs=6, validation_data=validation_gen)
# %%
test_pred = []
first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_pred.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_pred = scaler.inverse_transform(test_pred)
test["LSTMpred"] = true_pred
test.plot(figsize=(12, 10))
# %%
# forecast

full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
generator = TimeseriesGenerator(
    data=scaled_full_data,
    targets=scaled_full_data,
    length=length,
    batch_size=batch_size,
)

model = Sequential()

model.add(LSTM(units=50, input_shape=(length, n_features)))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mse")

model.summary()


# %%
model.fit(generator, epochs=6)

# %%
forecast = []
first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
forecast_len = 50
for i in range(forecast_len):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

forecast = scaler.inverse_transform(forecast)
forecast_index = np.arange(50.1, 50.1 + forecast_len * 0.1, step=0.1)
# %%
plt.plot(df.index, df["Sine"])
plt.plot(forecast_index, forecast)
# %%
full_df = pd.concat(
    [df, pd.DataFrame(forecast, columns=["Sine"], index=forecast_index)]
)

plt.plot(full_df.index, full_df["Sine"])
# %%

# %%
