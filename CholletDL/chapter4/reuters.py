import tensorflow.keras as keras
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

# The reuters dataset is set of short newswires and their topics.

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
    [reverse_word_index.get(i-3, "?") for i in train_data[0]])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)


# via tensorflow
# from keras.utils import to_categorical
# y_train = to_catergorical(train_labels)
# y_test = to_catergorical(test_labels)

model = keras.Sequential([
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dense(units=46, activation="softmax")  # we have 46 categories
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


loss_values = history.history["loss"]
val_loss_values = history.history["val_loss"]
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss value")
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# generation predictions on new data
predictions = model.predict(x_test)
prediction_labels = np.argmax(predictions, axis=1)

acc = sum(prediction_labels[i] == test_labels[i]
          for i in range(len(prediction_labels)))/len(prediction_labels)
print(f"accuracy on test data is {acc:.2f}")
