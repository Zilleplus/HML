import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def accuracy_for_learning_rate(learning_rate):
    print("Training model with learning rate={0}".format(learning_rate))
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=15, validation_split=0.2, verbose=1)
    [loss, accuracy] = model.test_on_batch(x=x_test,
                                           y=y_test)
    return (loss, accuracy)


learning_rates = []
accuracies = []
losses = []

for i in range(3, 15):
    learning_rate = 2**-i
    (loss, accuracy) = accuracy_for_learning_rate(learning_rate)
    learning_rates.append(learning_rate)
    accuracies.append(accuracy)
    losses.append(loss)

plt.semilogx(learning_rates, losses)
plt.show()

# The learning rate is optimal around 1e-3
