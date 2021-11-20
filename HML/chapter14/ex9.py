import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import math

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train = np.expand_dims(x_train, -1)
y_train = np.expand_dims(y_train, -1)

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_test = np.expand_dims(x_test, -1)
y_test = np.expand_dims(y_test, -1)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 7,
                        activation="relu",
                        padding="same",
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
                        activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters=128, kernel_size=3, strides=2,
                        activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])


class PrintOnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print(model.optimizer.iterations.numpy(),
              model.optimizer._decayed_lr(tf.float32).numpy())


batch_size = 32
earlystopping_cb = keras.callbacks.EarlyStopping(patience=6)
tensorboard_cb = keras.callbacks.TensorBoard("logsEx9")
callbacks = [earlystopping_cb, tensorboard_cb, PrintOnCallback()]
# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# We have 60 000 samples in the training set
# using a batch size of 32, the number of steps in 1 epoch is:
# math.ceil(60000/32)=1875 which is about 2000 steps per epoch.
ed_scheduler = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=1e-2,
     decay_steps=math.ceil(len(x_train)/batch_size),
     decay_rate=0.1,
     staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=ed_scheduler)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

start_time = time.time()
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=100,
    validation_split=0.2,
    verbose=1,
    callbacks=callbacks
)
end_time = time.time()
print("total learning time=" + str((end_time - start_time)/60) + " [minutes]")

[loss, accuracy] = model.test_on_batch(x=x_test, y=y_test)
print("loss=" + str(loss) + " accuracy=" + str(accuracy))
