# %%
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# import seaborn as sns
# import plotly.express as px

# from sklearn.preprocessing import MinMaxScaler
# %%
plt.rcParams["figure.figsize"] = (12, 8)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
X_train = X_train / 255
X_test = X_test / 255
# %%
encoder = Sequential()

encoder.add(Flatten(input_shape=[28, 28]))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=50, activation="relu"))
encoder.add(Dense(units=25, activation="relu"))
encoder.add(Dense(units=12, activation="relu"))
encoder.add(Dense(units=6, activation="relu"))
encoder.add(Dense(units=3, activation="relu"))


decoder = Sequential()

encoder.add(Dense(units=6, input_shape=[3], activation="relu"))
encoder.add(Dense(units=12, activation="relu"))
encoder.add(Dense(units=25, activation="relu"))
encoder.add(Dense(units=50, activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=784, activation="sigmoid"))

decoder.add(Reshape([28, 28]))


autoencoder = Sequential([encoder, decoder])

autoencoder.compile(
    loss="binary_crossentropy", optimizer=SGD(lr=2.5), metrics=["accuracy"]
)

autoencoder.summary()
# %%
encoder = Sequential()

encoder.add(Flatten(input_shape=[28, 28]))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=50, activation="relu"))
encoder.add(Dense(units=25, activation="relu"))

decoder = Sequential()
encoder.add(Dense(units=50, input_shape=[25], activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=784, activation="sigmoid"))

decoder.add(Reshape([28, 28]))


autoencoder = Sequential([encoder, decoder])

autoencoder.compile(
    loss="binary_crossentropy", optimizer=SGD(lr=2.5), metrics=["accuracy"]
)

autoencoder.summary()

# %%
autoencoder.fit(x=X_train, y=X_train, epochs=7)
# %%
passed_im = autoencoder.predict(X_test[:10])


def show_2(a, b, a_t="Original", b_t="Reconstruction"):
    f, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(a)
    ax[0].title.set_text(a_t)
    ax[1].imshow(b)
    ax[1].title.set_text(b_t)
    plt.show()


n = 3

show_2(X_test[n], passed_im[n])

# %%
#  Noise removal

sample = GaussianNoise(stddev=0.2)
noisey = sample(X_test[:10], training=True)
# %%
n = 0

show_2(X_test[n], noisey[n], b_t="Noisey")
# %%
tf.random.set_seed(101)
np.random.seed(101)

encoder = Sequential()
encoder.add(Flatten(input_shape=[28, 28]))
encoder.add(GaussianNoise(stddev=0.2))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=50, activation="relu"))
encoder.add(Dense(units=25, activation="relu"))

decoder = Sequential()
encoder.add(Dense(units=50, input_shape=[25], activation="relu"))
encoder.add(Dense(units=98, activation="relu"))
encoder.add(Dense(units=196, activation="relu"))
encoder.add(Dense(units=392, activation="relu"))
encoder.add(Dense(units=784, activation="sigmoid"))
decoder.add(Reshape([28, 28]))

noise_remover = Sequential([encoder, decoder])

noise_remover.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

autoencoder.summary()
# %%
noise_remover.fit(x=X_train, y=X_train, epochs=8)
# %%
ten_noisey_images = sample(X_test[:10], training=True)
denoised = noise_remover(ten_noisey_images)
# %%


def show_3(a, b, c):
    f, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(a)
    ax[0].title.set_text("Original")
    ax[1].imshow(b)
    ax[1].title.set_text("Noisey")
    ax[2].imshow(c)
    ax[2].title.set_text("Denoised")
    plt.show()


# %%
show_3(X_test[n], ten_noisey_images[n], denoised[n])
# %%
