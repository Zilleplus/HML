from pickletools import optimize
from numpy import gradient
import tensorflow as tf # type: ignore
import tensorflow.keras as keras # type: ignore
import tensorflow.keras.layers as layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore


def get_mnist_model():
    # define the model using the functional api
    inputs = keras.Input(shape=(28 * 28))
    features = layers.Dense(units=512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(units=10, activation="softmax")(features)

    model = keras.Model(inputs, outputs)
    return model


(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28*28)) \
    .astype("float32") / 255
test_images = test_images.reshape((test_images.shape[0], 28*28)) \
    .astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets, = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)

        loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [loss_tracker]

inputs = keras.Input(shape=(28 * 28))
features = layers.Dense(units=512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(units=10, activation="softmax")(features)

model = CustomModel(inputs, outputs)

model.compile(optimizer=keras.optimizers.RMSprop())
model.fit(train_images, train_labels, epochs=3)