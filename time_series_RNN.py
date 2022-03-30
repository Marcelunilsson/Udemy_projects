# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

plt.rcParams["figure.figsize"] = (12, 8)

# %%
# Economic data fred

df = pd.read_csv("../DATA/RSCCASN.csv", parse_dates=True, index_col="DATE")
df.columns = ["Sales"]
df.plot()
test_size = 18  # 1.5 years
test_index = len(df) - test_size
train = df.iloc[:test_index]
test = df.iloc[test_index:]

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

length = 12  # one year
batch_size = 1

generator = TimeseriesGenerator(
    data=train_scaled, targets=train_scaled, length=length, batch_size=batch_size
)


# %%
n_features = 1
model = Sequential()

model.add(layer=LSTM(units=128, activation="relu", input_shape=(length, n_features)))

model.add(layer=Dense(units=1))
model.compile(optimizer="adam", loss="mse")

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=4)

validation_gen = TimeseriesGenerator(
    data=test_scaled, targets=test_scaled, length=length, batch_size=batch_size
)
# %%
model.fit(generator, epochs=20, validation_data=validation_gen, callbacks=[early_stop])
# %%
losses = pd.DataFrame(model.history.history)
losses.plot()
# %%
pred = []

first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]

    pred.append(current_pred)
    current_batch = np.append(
        arr=current_batch[:, 1:, :], values=[[current_pred]], axis=1
    )

# %%
true_pred = scaler.inverse_transform(pred)
test["pred"] = true_pred
# %%
test.plot()
# %%
# forecast

full_scaler = MinMaxScaler()
full_data_scaled = full_scaler.fit_transform(df)

length = 12
generator = TimeseriesGenerator(
    data=full_data_scaled,
    targets=full_data_scaled,
    length=length,
    batch_size=batch_size,
)
# %%
model = Sequential()


model.add(layer=LSTM(units=128, activation="relu", input_shape=(length, n_features)))

model.add(layer=Dense(units=1))

model.compile(optimizer="adam", loss="mse")

model.summary()

model.fit(
    generator,
    epochs=13,
)
# %%
forecast = []
forecast_length = 120

first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(forecast_length):
    current_pred = model.predict(current_batch)[0]

    forecast.append(current_pred)
    current_batch = np.append(
        arr=current_batch[:, 1:, :], values=[[current_pred]], axis=1
    )
# %%
forecast = scaler.inverse_transform(forecast)
forecast_index = pd.date_range(
    start="2019-11-01", periods=forecast_length, freq="MS"
)  # MS - monthly start


forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=["Forecast"])

ax = df.plot()
forecast_df.plot(ax=ax)
# %%
ax = df.plot()
forecast_df.plot(ax=ax)
plt.xlim("2018-01-01", "2020-12-01")
# %%
