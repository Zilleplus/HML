import tensorflow as tf # type: ignore
import tensorflow.keras as keras # type: ignore
import tensorflow.keras.layers as layers # type: ignore
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


dataset_dir = Path(__file__).parent.parent/"cats_vs_dogs_small"
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


conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3))

conv_base.summary()

def get_all_features_and_labels(dataset):
    all_features = []
    all_lables = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_lables.append(labels)

    return np.concatenate(all_features), np.concatenate(all_lables)

train_features, train_labels = get_all_features_and_labels(train_dataset)
val_features, val_labels = get_all_features_and_labels(validation_dataset)
test_features, test_labels = get_all_features_and_labels(test_dataset)

inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics = ["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss")]

history = model.fit(
    x=train_features, y=train_labels,
    epochs=20,
    validation_data=(val_features, val_labels),
    callbacks=callbacks)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Train accuracy")
plt.plot(epochs, val_acc, "b", label="validation accuracy")
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