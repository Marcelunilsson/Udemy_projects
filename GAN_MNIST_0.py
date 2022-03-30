# %%
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential

plt.rcParams["figure.figsize"] = (12, 8)
# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Scale images -1.0 to 1.0
X_train = X_train / 255
X_train = X_train.reshape(-1, 28, 28, 1) * 2.0 - 1


zeros = X_train[y_train == 0]

# %%
coding_size = 128
generator = Sequential()
generator.add(Dense(units=7 * 7 * 128, activation="relu", input_shape=[coding_size]))
generator.add(Reshape([7, 7, 128]))
generator.add(BatchNormalization())
generator.add(
    Conv2DTranspose(
        filters=64, kernel_size=5, strides=2, padding="same", activation="relu"
    )
)
generator.add(BatchNormalization())
generator.add(
    Conv2DTranspose(
        filters=1, kernel_size=5, strides=2, padding="same", activation="tanh"
    )
)


discriminator = Sequential()
discriminator.add(
    Conv2D(
        filters=64,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=LeakyReLU(0.3),
        input_shape=[28, 28, 1],
    )
)
discriminator.add(Dropout(rate=0.5))
discriminator.add(
    Conv2D(
        filters=128, kernel_size=5, strides=2, padding="same", activation=LeakyReLU(0.3)
    )
)
discriminator.add(Dropout(rate=0.5))
discriminator.add(Flatten())
discriminator.add(Dense(units=1, activation="sigmoid"))

discriminator.compile(loss="binary_crossentropy", optimizer="adam")
discriminator.trainable = False

GAN = Sequential([generator, discriminator])

GAN.compile(loss="binary_crossentropy", optimizer="adam")
# %%
batch_size = 32
epochs = 20
my_data = zeros
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(
    buffer_size=1
)
generator, discriminator = GAN.layers
# %%
for epoch in range(epochs):
    print(f"Currently on Epoch: {epoch+1}")
    i = 0
    for X_batch in dataset:
        i += 1
        if i % 20 == 0:
            print(f"\t Currently on batch number {i} of {len(my_data)//batch_size}")

        # Discriminator

        noise = tf.random.normal(shape=[batch_size, coding_size])
        gen_images = generator(noise)

        X_fake_vs_real = tf.concat(
            [gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis=0
        )
        y1 = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)

        discriminator.trainable = True
        discriminator.train_on_batch(x=X_fake_vs_real, y=y1)

        noise = tf.random.normal(shape=[batch_size, coding_size])
        y2 = tf.constant([[1.0]] * batch_size)
        discriminator.trainable = False
        GAN.train_on_batch(x=noise, y=y2)


# %%

noise = tf.random.normal(shape=[10, coding_size])
plt.imshow(noise)
plt.show()

images = generator(noise)
# %%
for image in images:
    plt.imshow(image.numpy().reshape(28, 28))
    plt.show()

# %%
