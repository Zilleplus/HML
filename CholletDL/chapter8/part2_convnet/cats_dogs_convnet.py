from pathlib import Path
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])


dataset_dir = Path(__file__).parent.parent/"cats_vs_dogs_small"
# dataset_dir = Path("../cats_vs_dogs_small")  # to use this with slime

train_dataset = image_dataset_from_directory(
    directory=dataset_dir / "train",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    directory=dataset_dir/"validation",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32)
test_dataset = image_dataset_from_directory(
    directory=dataset_dir/"test",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scrach.keras",
        save_best_only=True,
        monitor="val_loss")]

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=callbacks)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Train accuracy")
plt.plot(epochs, val_accuracy, "b", label="validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('train_val_acc.png', bbox_inches='tight')
plt.figure()
plt.plot(epochs, loss, "bo", label="Train loss")
plt.plot(epochs, val_loss, "b", label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('train_val_loss.png', bbox_inches='tight')
plt.show()
