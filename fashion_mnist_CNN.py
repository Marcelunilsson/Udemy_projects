# %%
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# from tensorflow.keras.layers import Dropout

# %%
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))


y_train_c = to_categorical(y_train, num_classes=10)
y_test_c = to_categorical(y_test, num_classes=10)
image_shape = (28, 28)

image_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1 / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)


image_gen.fit(x_train)
# %%

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        # kernel_initializer='he_uniform',
        input_shape=(28, 28, 1),
        activation="relu",
        padding="same",
    )
)
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        # kernel_initializer='he_uniform',
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=64,
#                  kernel_size=(3, 3),
#                  kernel_initializer='he_uniform',
#                  activation='relu',
#                  padding='same'))
# model.add(Conv2D(filters=64,
#                  kernel_size=(3, 3),
#                  kernel_initializer='he_uniform',
#                  activation='relu',
#                  padding='same'))
# model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation="relu"))
# model.add(Dropout(rate=.3))

model.add(Dense(units=10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
# %%
batch_size = 32
steps = len(x_train) / batch_size
train_image_gen = image_gen.flow(x_train, y_train_c, batch_size=batch_size)

test_image_gen = image_gen.flow(
    x_test,
    y_test_c,
    batch_size=batch_size,
)


result = model.fit(
    train_image_gen, steps_per_epoch=steps, epochs=10, validation_data=test_image_gen
)

# %%
y_pred = model.predict_classes(x_test)

losses = pd.DataFrame(result.history)

losses[["accuracy", "val_accuracy"]].plot()

evaluation = model.evaluate(x_test, y_test_c, verbose=0)

print(f"{ model.metrics_names }\n { evaluation }")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

cmd = ConfusionMatrixDisplay(cm)

cmd.plot()

# %%
