import tensorflow as tf # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import tensorflow.keras.layers as layers # type: ignore
import tensorflow.keras as keras # type: ignore


def get_mnist_model():
    inputs = keras.Input(shape=(28*28,))
    features = layers.Dense(units=512, activation="relu")(inputs)
    features = layers.Dropout(rate=0.5)(features)
    outputs = layers.Dense(units=10, activation="softmax")(features)

    model = keras.Model(inputs, outputs)
    return model


(images, labels), (test_image, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_image.reshape((10000, 28*28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]


model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=2,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)