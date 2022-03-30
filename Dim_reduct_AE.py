# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D


# %%
plt.rcParams["figure.figsize"] = (12, 8)

data = make_blobs(
    n_samples=300, n_features=2, centers=2, cluster_std=1.0, random_state=101
)

X, y = data
np.random.seed(seed=101)
z_noise = pd.Series(np.random.normal(size=len(X)))
feat = pd.DataFrame(X)
feat = pd.concat([feat, z_noise], axis=1)
feat.columns = ["X1", "X2", "X3"]


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(feat.X1, feat.X2, feat.X3, c=y)
# %%
fig = px.scatter_3d(feat, x="X1", y="X2", z="X3", color=y)
fig.show()
# %%

encoder = Sequential()

encoder.add(Dense(units=2, activation="relu", input_shape=[3]))

decoder = Sequential()

decoder.add(Dense(units=3, activation="relu", input_shape=[2]))

autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=SGD(lr=1.5))


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(feat)

autoencoder.summary()
# %%
autoencoder.fit(x=scaled_data, y=scaled_data, epochs=5)
# %%
encoded_2dim = encoder.predict(scaled_data)
px.scatter(x=encoded_2dim[:, 0], y=scaled_data[:, 1], color=y)

# %%
fig = px.scatter_3d(feat, x="X1", y="X2", z="X3", color=y)
