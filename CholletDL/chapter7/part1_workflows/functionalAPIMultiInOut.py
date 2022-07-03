import tensorflow as tf # type: ignore
import tensorflow.keras as keras # type: ignore
import numpy as np

tf.enable_eager_execution()

vocabulary_size = 1000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags, ), name="tags")

features = keras.layers.Concatenate()([title, text_body, tags])
features = keras.layers.Dense(units=64, activation="relu")(features)

priority = keras.layers.Dense(
    units=1, activation="sigmoid", name="priority")(features)
department = keras.layers.Dense(
    units=num_departments, activation="softmax", name="department")(features)

model = keras.Model(
    inputs=[title, text_body, tags], outputs=[priority, department])

# create some dummy data

num_samples = 1280

titel_data = np.random.randint(low=0, high=2,
                               size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(low=0, high=2,
                                   size=(num_samples, vocabulary_size))
tags_data = np.random.randint(low=0, high=2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(low=0, high=2,
                                    size=(num_samples, num_departments))

model.compile(
              optimizer="rmsprop",
              # priority is float
              # department is category
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])

model.fit([titel_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)

priority_preds, department_preds = model.predict(
    [titel_data, text_body_data, tags_data])

keras.utils.plot_model(model, "ticket_classifier.png")