# %%
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# %%
# make categorical and scale data
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)

x_train = x_train / 255
x_test = x_test / 255

# %%
# Reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# %%
# Model creation

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        input_shape=(28, 28, 1),
        activation="relu",
        padding="same",
    )
)
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(.05))

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        input_shape=(32, 32, 3),
        activation="relu",
        padding="same",
    )
)
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(.1))

model.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        input_shape=(32, 32, 3),
        activation="relu",
        padding="same",
    )
)
model.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(.15))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10, activation="softmax"))
opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# Model summary
model.summary()
# %%
# Train model
early_stop = EarlyStopping(monitor="val_loss", patience=1)
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.3,
)
datagen.fit(x_train)
bs = 128
steps = 0.7 * len(x_train) / bs

# hist = model.fit(x_train,
#                 y_cat_train,
#                 epochs=10,
#                 validation_data=(x_test, y_cat_test),
#                 callbacks=[early_stop])

hist = model.fit(
    datagen.flow(x_train, y_cat_train, batch_size=bs, subset="training"),
    steps_per_epoch=steps,
    epochs=100,
    validation_data=datagen.flow(
        x_train, y_cat_train, batch_size=bs, subset="validation"
    ),
)

# %%
# Evaluate model

y_pred = model.predict_classes(x_test)

losses = pd.DataFrame(hist.history)

losses[["accuracy", "val_accuracy"]].plot()

losses[["loss", "val_loss"]].plot()

evaluation = model.evaluate(x_test, y_cat_test, verbose=0)

print(f"{ model.metrics_names }\n { evaluation }")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

cmd = ConfusionMatrixDisplay(cm)

cmd.plot()


# %%

# %%
