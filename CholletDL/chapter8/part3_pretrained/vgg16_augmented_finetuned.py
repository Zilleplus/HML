import tensorflow.keras as keras  # type: ignore
import tensorflow.keras.layers as layers  # type: ignore
from pathlib import Path
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

batch_size = 32

dataset_dir = Path(__file__).parent.parent/"cats_vs_dogs_small"
# dataset_dir = Path("../cats_vs_dogs_small")  # to use this with slime

train_dataset = image_dataset_from_directory(
    directory=dataset_dir / "train",
    image_size=(180, 180),  # the default is 256*256
    batch_size=batch_size).prefetch(2)
validation_dataset = image_dataset_from_directory(
    directory=dataset_dir/"validation",
    image_size=(180, 180),  # the default is 256*256
    batch_size=batch_size)
test_dataset = image_dataset_from_directory(
    directory=dataset_dir/"test",
    image_size=(180, 180),  # the default is 256*256
    batch_size=batch_size)

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3))
# Make sure we don't spend time adjusting the pretained weights:
conv_base.trainable = False

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_augmented.keras",
        save_best_only=True,
        monitor="val_loss"),
    # If no improvement after 3 epochs -> early stop
    keras.callbacks.EarlyStopping(patience=3)
]

history = model.fit(
    x=train_dataset,
    epochs=50,
    validation_data=validation_dataset,
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
plt.savefig('train_val_acc_augmented_before_tuned.png', bbox_inches='tight')
plt.figure()
plt.plot(epochs, loss, "bo", label="Train loss")
plt.plot(epochs, val_loss, "b", label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('train_val_loss_augmented_before_tuned.png', bbox_inches='tight')
plt.show()

conv_base.trainable = True
for layer in conv_base.layers[:-4]:  # all the layers but the last 4
    layer.trainable = False

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_augmented.keras",
        save_best_only=True,
        monitor="val_loss"),
    # If no improvement after 3 epochs -> early stop
    keras.callbacks.EarlyStopping(patience=3)
    ]

history = model.fit(
    x=train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, "bo", label="Train accuracy")
plt.plot(epochs, val_acc, "b", label="validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('train_val_acc_augmented_tuned.png', bbox_inches='tight')
plt.figure()
plt.plot(epochs, loss, "bo", label="Train loss")
plt.plot(epochs, val_loss, "b", label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('train_val_loss_augmented_tuned.png', bbox_inches='tight')
plt.show()
