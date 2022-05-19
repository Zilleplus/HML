import tensorflow.keras as keras

model = keras.Sequential([
    keras.layers.Dense(1)])

model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])

model.compile(optimizer=keras.optimizers.RMSprop)
