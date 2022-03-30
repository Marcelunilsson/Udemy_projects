# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import seaborn as sns
# from matplotlib.image import imread

plt.rcParams["figure.figsize"] = (12, 10)
# %%
data_dir = "_Data/cell_images"
test_path = data_dir + "/test"
train_path = data_dir + "/train"
para = "/parasitized/"
uninf = "/uninfected/"
image_shape = (130, 130, 3)


# %%
image_gen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
# %%
model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=image_shape,
        activation="relu",
        padding="same",
    )
)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        input_shape=image_shape,
        activation="relu",
        padding="same",
    )
)
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))


# model.add(Conv2D(filters=128,
#                  kernel_size=(3,3),
#                  input_shape=image_shape,
#                  activation='relu',
#                  padding='same'))
# model.add(Conv2D(filters=128,
#                  kernel_size=(3,3),
#                  activation='relu',
#                  padding='same'))
# model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation="relu"))
model.add(Dropout(rate=0.5))

model.add(Dense(units=1, activation="sigmoid"))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
# %%
es = EarlyStopping(monitor="val_loss", patience=2)
batch_size = 16
train_image_gen = image_gen.flow_from_directory(
    train_path,
    target_size=image_shape[:2],
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="binary",
)

test_image_gen = image_gen.flow_from_directory(
    test_path,
    target_size=image_shape[:2],
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)
result = model.fit(
    train_image_gen, epochs=20, validation_data=test_image_gen, callbacks=[]
)
# %%
pred = model.predict(test_image_gen)
losses = pd.DataFrame(result.history)
losses[["accuracy", "val_accuracy"]].plot()
losses[["loss", "val_loss"]].plot()
y_pred = pred > 0.5
y_test = test_image_gen.classes
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot()
print(classification_report(y_test, y_pred))
# %%

# Predict new
para_cell_path = train_path + para + os.listdir(train_path + para)[15]
para_cell_im = image.load_img(para_cell_path, target_size=image_shape)

my_im_arr = image.img_to_array(para_cell_im)

my_im_arr = np.reshape(my_im_arr, (1, 130, 130, 3))

new_pred = ((model.predict(my_im_arr) > 0.5) * 1)[0][0]

print(
    f"Prediction: {new_pred} \n True: parasitized \nclasses: {train_image_gen.class_indices}"
)


# %%
