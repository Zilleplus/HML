import tensorflow.keras as keras # type: ignore
import tensorflow.keras.layers as layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from pathlib import Path # type: ignore


def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28))
    features = layers.Dense(units=512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(units=10, activation="softmax")(features)

    model = keras.Model(inputs, outputs)
    return model


(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28*28)).astype("float32") / 255
test_images = test_images.reshape((test_images.shape[0], 28*28)) \
    .astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]


model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

log_dir = Path("./tensorboard_log").absolute()
callback_list = [keras.callbacks.TensorBoard(log_dir=log_dir)]
model.fit(train_images, train_labels,
          epochs=10,
          callbacks=callback_list,
          validation_data=(val_images, val_labels))
