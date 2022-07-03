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


model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# We won't use this:
# model.fit(train_images, train_labels,
#           epochs=10,
#           validation_data=(val_images, val_labels))


loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()


def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()
    return logs


def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()


training_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
training_dataset = training_dataset.batch(32)
epochs = 3
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(inputs_batch, targets_batch)
    print(f"Results at the end of epoch {epoch}")
    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")


# uncomment this to disable eagerly execution, and enable the computational graph. 
# The computational graph is faster, but you can't debug it.
# @tf.function # computational graph
def test_step(inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)
    logs["val_loss"] = loss_tracking_metric.result()
    return logs


val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()
for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)
    print("Evaluate results")
    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")
