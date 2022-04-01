# %%
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

plt.rcParams["figure.figsize"] = (12, 8)


# %%
df = pd.read_csv("_Data/UK_foods.csv", index_col="Unnamed: 0")
df = df.T
# %%
sns.heatmap(data=df.T)
# %%


encoder = Sequential()

encoder.add(Dense(units=8, activation="relu", input_shape=[17]))
encoder.add(Dense(units=4, activation="relu"))
encoder.add(Dense(units=2, activation="relu"))

decoder = Sequential()

decoder.add(Dense(units=4, activation="relu", input_shape=[2]))
decoder.add(Dense(units=8, activation="relu"))
decoder.add(Dense(units=17, activation="relu"))

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss="mse", optimizer=SGD(lr=1.5))
autoencoder.summary()
# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

autoencoder.fit(x=scaled_data, y=scaled_data, epochs=15)

red_dim = encoder.predict(scaled_data)
result = pd.DataFrame(data=red_dim, index=df.index, columns=["C1", "C2"])
result = result.reset_index()
sns.scatterplot(data=result, x="C1", y="C2", hue="index")

px.scatter(data_frame=result, x="C1", y="C2", color="index")
# %%
result.head()
# %%
