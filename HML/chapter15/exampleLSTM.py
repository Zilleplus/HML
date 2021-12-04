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
series = generate_time_series(batch_size=n_samples, n_steps=n_steps + 10)
x_train, y_train_single = series[:7000, :n_steps], series[:7000, -1]
x_valid, y_valid_single = series[7000:9000, :n_steps], series[7000:9000, -1]
x_test, y_test_single = series[9000:, :n_steps], series[9000:, -1]

Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10+1):
    Y[:, :, step_ahead-1] = series[:, step_ahead:(step_ahead + n_steps), 0]
y_train = Y[:7000]
y_valid = Y[7000:9000]
y_test = Y[9000:]

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[50, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1)),
])


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse",
              optimizer=optimizer, metrics=[last_time_step_mse])
model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_valid, y_valid),
    epochs=20)
[loss, accuracy] = model.test_on_batch(x=x_test, y=y_test)
print("loss=" + str(loss) + " accuracy=" + str(accuracy))
