import tensorflow as tf
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

learning_rate = 1e-3
print("Training model with learning rate={0}".format(learning_rate))
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# -> replace with with manual stuff
# optimizer = keras.optimizers.SGD(lr=learning_rate)
# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer=optimizer,
#               metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=15, validation_split=0.2, verbose=1)
# -> end of: replace with with manual stuff
# [loss, accuracy] = model.test_on_batch(x=x_test,
#                                        y=y_test)

n_epochs = 15
batch_size = 256
n_steps = len(x_train) // batch_size
optimizer = keras.optimizers.SGD(lr=learning_rate)#keras.optimizers.Nadam(lr=learning_rate)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]


def one_hot_encoding(x):
    n_vals = 10
    return np.eye(n_vals)[x]


def one_hot_decoding(x):
    return [np.argmax(el) for el in x]


def random_batch(x, y, batch_size=32):
    idx = np.random.randint(len(x), size=batch_size)
    return x[idx], y[idx]


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
        for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics + "\n", end=end)


for epoch in range(1, n_epochs + 1):
    print("Epoch{}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        x_batch, y_batch = random_batch(x_train, y_train)
        y_batch = one_hot_encoding(y_batch)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        _ = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        _ = mean_loss(loss)
        for metric in metrics:
            _ = metric(y_batch, y_pred)
    print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()
y_pred = model.predict(x_test)
y_pred_decoded = one_hot_decoding(y_pred)
res = y_pred_decoded - y_test
acc = sum([1 if x != 0 else 0 for x in res]) / len(y_test)
print("Accuracy = {}".format(acc))

# Loss = 0.213
# Accuracy 0.952
