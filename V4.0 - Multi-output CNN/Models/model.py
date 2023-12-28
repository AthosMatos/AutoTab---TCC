import tensorflow as tf
import numpy as np
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
    Resizing,
)
from keras.losses import (
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
)
from keras.callbacks import Callback
import os
from keras.callbacks import LearningRateScheduler

train_ds, unique_labels = (
    np.load("/content/drive/MyDrive/AutoTAB/notes_np_cqt_44.1k/train_ds-6out.npz"),
    np.load("/content/drive/MyDrive/AutoTAB/all_labels.npy"),
)

train_x = train_ds["x"]
# train_y = train_ds["y"]

train_y = []
# fill train_y with all the values from train_ds["y"]
for i in range(len(unique_labels)):
    train_y.append(train_ds["y"][:, i])

input_shape = (train_x.shape[1], train_x.shape[2], 1)
num_labels = len(unique_labels)

print("")

print(f"Training with {len(train_x)} files")
print("")


inputs = Input(shape=input_shape)
x = Conv2D(32, 3, padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(256, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)

x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Dropout(0.5)(x)
""" 
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x) """
# outs = Dense(num_labels, activation="sigmoid")(x)

outs = []

for _ in range(num_labels):
    outs.append(Dense(1, activation="sigmoid")(x))

model = tf.keras.Model(inputs, outs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=BinaryCrossentropy(),  # BinaryFocalCrossentropy(),
    metrics=["accuracy"],
)


class CustomCallback(Callback):
    def __init__(self, num_dense_layers):
        super(CustomCallback, self).__init__()
        self.num_dense_layers = num_dense_layers
        self.avg_precision = 0
        self.avg_loss = 0
        # self.avg_val_precision = 0
        # self.avg_val_loss = 0

    """ def on_batch_end(self, batch, logs=None):
        #print a progress bar at each 10 files processed

       
        #os.system("cls")
        print(f"Processed {batch*128} / {len(train_x)} files") """

    def on_epoch_end(self, epoch, logs=None):
        # Extract precision for each dense output from the logs
        denses_precision = 0
        denses_loss = 0
        denses_val_precision = 0
        denses_val_loss = 0

        for i in range(1, self.num_dense_layers):
            train_loss = logs[f"dense_{i}_loss"]
            train_accuracy = logs[f"dense_{i}_accuracy"]
            """ val_loss = logs[f"val_dense_{i}_loss"]
            val_accuracy = logs[f"val_dense_{i}_accuracy"]
 """
            # Update total precision and loss
            denses_precision += train_accuracy
            denses_loss += train_loss

            """  # Update total validation precision and loss
            denses_val_precision += val_accuracy
            denses_val_loss += val_loss """

        # Print average training and validation metrics
        self.avg_precision = denses_precision / self.num_dense_layers
        self.avg_loss = denses_loss / self.num_dense_layers
        """ self.avg_val_precision = denses_val_precision / self.num_dense_layers
        self.avg_val_loss = denses_val_loss / self.num_dense_layers """

        print(
            f"|| Epoch {epoch + 1} ||  Avg Precision: Train {self.avg_precision} Val {self.avg_val_precision}| Avg Loss: Train {self.avg_loss} Val {self.avg_val_loss}"
        )
        print()

    def on_train_end(self, logs=None):
        # Print the overall average precision and final average loss at the end of training
        print(
            f"OverallAvg Precision: Train {self.avg_precision} Val {self.avg_val_precision}| Avg Loss: Train {self.avg_loss} Val {self.avg_val_loss}"
        )


custom_callback = CustomCallback(num_labels)


def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9
    return lr


lr_schedule = LearningRateScheduler(lr_scheduler)

callbacks = [
    custom_callback,
    lr_schedule
    # tf.keras.callbacks.ModelCheckpoint("/content/drive/MyDrive/AutoTAB/bestModel_save_at_{epoch}.h5"),
    # tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
]


model.fit(
    train_x,
    train_y,
    # validation_split=0.2,
    batch_size=128,
    epochs=28,
    verbose=1,
    # callbacks=callbacks
)

# save the model
model.save("/content/drive/MyDrive/AutoTAB/Models/model_notes.h5")
