from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore

dataset_dir = Path(__file__).parent.parent / "chapter8" / "cats_vs_dogs_small"

# to use this with slime
# dataset_dir = Path("../chapter8/cats_vs_dogs_small")

train_dataset = image_dataset_from_directory(
    directory=dataset_dir / "train",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32,
)
validation_dataset = image_dataset_from_directory(
    directory=dataset_dir / "validation",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32,
)
test_dataset = image_dataset_from_directory(
    directory=dataset_dir / "test",
    image_size=(180, 180),  # the default is 256*256
    batch_size=32,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)


def get_model():
    inputs = keras.Input(shape=(180, 180, 3))

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

    for size in (32, 64, 128, 256, 512):
        residual = x

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(
            kernel_size=size, filters=1, strides=2, padding="same", use_bias=False
        )(residual)
        x = layers.add([x, residual])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


model = get_model()
model.summary()

# binary_crossentropy: 2 classes [cat; dog]
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_base_only=True),
    keras.callbacks.EarlyStopping(patience=3),
]

history = model.fit(
    train_dataset, epochs=50, callbacks=callbacks, validation_data=validation_dataset
)

loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(loss, "bo", label="Training loss")
plt.plot(val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
