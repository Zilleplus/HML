import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
print("using tensorflow version: " + tf.__version__)


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2, = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offset2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
n_samples = 10000
series = generate_time_series(batch_size=n_samples, n_steps=n_steps + 1)
x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
x_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# Every sample is a serie of 50 values (x) with 1 prediction (y) at the end.
plt.plot(x_train[0])
plt.show()


def compile_and_fit(model, learning_rate=1e-3, epochs=20):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse",
                  optimizer=optimizer, metrics=["mse"])
    model.fit(x_train, y_train, epochs=epochs)
    [loss, accuracy] = model.test_on_batch(x=x_test, y=y_test)
    print("loss=" + str(loss) + " accuracy=" + str(accuracy))
    return (loss, accuracy)


# Native forcasting with linear regression model.
model_linear_regression = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])
linear_loss, linear_accuracy = compile_and_fit(model_linear_regression)

f_x = np.expand_dims(x_test[0], axis=0)  # batch of 1 dimension
f_y = y_test[0]
f_pred = model_linear_regression.predict(f_x)
print((f_pred - f_y)*(f_pred - f_y))

model_single_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1]),
])
single_rnn_loss, single_rnn_accuracy = compile_and_fit(model_single_rnn)
# single rnn weights:
# single_rnn_layer = model_single_rnn.layers[0]
# single_rnn_layer.weights[0] => input matrix W_x
# single_rnn_layer.weights[1] => recurrent_matrix W_y
# single_rnn_layer.weights[2] => bias matrix


model_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(1),
])
rnn_loss, rnn_accuracy = compile_and_fit(model_rnn,
                                         learning_rate=1e-3,
                                         epochs=20)
