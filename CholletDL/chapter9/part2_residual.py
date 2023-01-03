from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore


def example_no_maxpooling():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    residual = x
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                      padding="same")(x)
    # Because padding="same" -> no downsample due to padding.
    # The residual does however only have 32 channels.
    # -> Use Conv2D to get 64 channels.
    residual = layers.Conv2D(filters=64, kernel_size=1)(residual)
    x = layers.add([x, residual])


example_no_maxpooling()


def example_maxpooling():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    residual = x
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                      padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2, padding="same")(x)
    # Because padding="same" -> no downsample due to padding.
    # Max-pool of pool_size=2 reduces the size of the filters.
    # Use the conv2d to downsample the residual.
    residual = layers.Conv2D(filters=64, kernel_size=1, strides=2)(residual)
    x = layers.add([x, residual])


example_maxpooling()


inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)


def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters=filters, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=filters, kernel_size=3,
                      activation="relu", padding="same")(x)

    if pooling:
        x = layers.MaxPooling2D(pool_size=2, padding="same")(x)
        residual = layers.Conv2D(
            filters=filters, kernel_size=1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters=filters, kernel_size=1)(residual)
    x = layers.add([x, residual])
    return x


x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)


x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(units=1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()