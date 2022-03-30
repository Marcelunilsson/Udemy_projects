import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# make categorical and scale data
y_cat_test = to_categorical(y_test)
y_cat_train = to_categorical(y_train)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Model creation

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        input_shape=(32, 32, 3),
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
# model.add(Dropout(.1))

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
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
# model.add(Dropout(.1))

model.add(Flatten())

model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(Dropout(0.3))

model.add(Dense(10, activation="softmax"))
opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Model summary
model.summary()
# Train model
early_stop = EarlyStopping(monitor="val_loss", patience=1)
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)
datagen.fit(x_train)
bs = 128
steps = 0.8 * len(x_train) / bs

hist = model.fit(
    datagen.flow(x_train, y_cat_train, batch_size=bs, subset="training"),
    steps_per_epoch=steps,
    epochs=150,
    validation_data=datagen.flow(
        x_train, y_cat_train, batch_size=bs, subset="validation"
    ),
)

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
