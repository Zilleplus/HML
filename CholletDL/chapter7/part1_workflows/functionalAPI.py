import tensorflow.keras as keras # type: ignore

inputs = keras.Input(shape=(3,), name="input_layer")
features = keras.layers.Dense(units=64, activation="relu", name="middel_layer")(inputs)
outputs = keras.layers.Dense(units=10, activation="softmax", name="ouputer_layer")(features)
model = keras.Model(inputs=inputs, outputs=outputs)


# Just like with seq-api, but this time we don't need tot call the build.
model.summery()