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

#model = keras.models.Sequential([
#    keras.layers.Conv2D(32, 7,
#                        activation="relu",
#                        padding="same",
#                        input_shape=(28, 28, 1)),
#    keras.layers.MaxPooling2D(2),
#    keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
#                        activation="relu", padding="same"),
#    keras.layers.MaxPooling2D(2),
#    keras.layers.Conv2D(filters=128, kernel_size=3, strides=2,
#                        activation="relu", padding="same"),
#    keras.layers.MaxPooling2D(2),
#    keras.layers.Flatten(),
#    keras.layers.Dense(128, activation="relu"),
#    keras.layers.Dropout(0.5),
#    keras.layers.Dense(10, activation="softmax"),
#])


input_model = keras.layers.input()
l1 = keras.layers.Conv2D(32, 7, activation="relu", padding="same", input_shape=(28, 28, 1))(input_model)
l2 = keras.layers.MaxPooling2D(2)(l1)
l3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(l2)
l4 = keras.layers.MaxPooling2D(2)(l3)
l5 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(l4)
l6 = keras.layers.MaxPooling2D(2)(l5)
l7 = keras.layers.Flatten()(l6)
l8 = keras.layers.Dense(128, activation="relu")(l7)
l9 = keras.layers.Dropout(0.5)(l8)
output = keras.layers.Dense(10, activation="softmax")(l9)
model = keras.Model(inputs=[input_model], outputs=output)


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_status(iteration, total, loss_metric, mean_abs_err, accuracy):
    print(str(iteration) + "/" + str(total) +
          " mean_abs_err: " + str(mean_abs_err.result().numpy()) +
          " loss metric: " + str(loss_metric.result().numpy()) +
          " accuracy: " + str(accuracy.result().numpy()))


mean_loss = keras.metrics.Mean()
mean_abs_error = keras.metrics.MeanAbsoluteError()
accuracy = keras.metrics.Accuracy()
optimizer = keras.optimizers.Nadam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy()
batch_size = 32
number_of_steps = len(x_train)//batch_size
number_of_epochs = 10
for epoch in range(number_of_epochs):
    print("Epoch " + str(epoch))
    for step in range(number_of_steps):
        x_batch, y_batch = random_batch(x_train, y_train)
        y_batch_one_hot = tf.one_hot(y_batch, depth=10, axis=1)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch_one_hot, y_pred))
            # Add things like regularization losses to the total loss
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        _ = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        _ = mean_loss(loss)
        _ = mean_abs_error(y_batch_one_hot, y_pred)
        predicted_labels = tf.argmax(input=y_pred, axis=1)
        accuracy.update_state(y_true=y_batch, y_pred=predicted_labels)
    print_status(len(y_train), len(y_train),
                 loss_metric=mean_loss,
                 mean_abs_err=mean_abs_error,
                 accuracy=accuracy)
    mean_loss.reset_state()
    mean_abs_error.reset_state()
    accuracy.reset_state()
