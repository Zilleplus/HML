import numpy as np
import tensorflow.keras as keras


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


def compile_and_fit(model):
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss="mse",
                  optimizer=optimizer, metrics=["mse"])
    model.fit(x_train, y_train, epochs=20)
    [loss, accuracy] = model.test_on_batch(x=x_test, y=y_test)
    print("loss=" + str(loss) + " accuracy=" + str(accuracy))


model_linear_regression = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])
compile_and_fit(model_linear_regression)


model_single_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1]),
])
compile_and_fit(model_single_rnn)


model_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[50, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(1),
])
compile_and_fit(model_rnn)
