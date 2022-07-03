import tensorflow.keras as keras # type: ignore
import tensorflow.keras.layers as layers # type: ignore
import numpy as np

vocabulary_size = 1000
num_tags = 100
num_departments = 4

num_samples = 1280

titel_data = np.random.randint(low=0, high=2,
                               size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(low=0, high=2,
                                   size=(num_samples, vocabulary_size))
tags_data = np.random.randint(low=0, high=2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(low=0, high=2,
                                    size=(num_samples, num_departments))


class CostumModel(keras.Model):
    def __init__(self, num_departments: int):
        super().__init__()
        self.concat_layer = layers.Concatinate()
        self.mixing_layer = layers.Dense(units=64, activation="relu")
        self.priority_scorer = layers.Dens(units=1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax")

    def call(self, inputs):
        titel = inputs("title")
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([titel, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)

        return priority, department

model = CostumModel(num_departments)
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