import tensorflow as tf
import tensorflow.keras as keras
import pathlib
import os


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%s")
    return os.path.join("logsEx10", run_id)


data_dir = pathlib.Path("/home/zilleplus/reduced_oxford_iit")
animals = list(data_dir.glob("*.jpg"))

batch_size = 16
data = keras.preprocessing.image_dataset_from_directory(
    str(data_dir),
    shuffle=True,
    batch_size=batch_size,
    seed=123,
    image_size=(224, 224))

test_data = data.take(2)
validate_data = data.skip(len(test_data)).take(2)
train_data = data.skip(len(test_data) + len(validate_data))

autotune = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(autotune)

number_of_batches_in_training_set = len(train_data)
s = 5*number_of_batches_in_training_set
print("decaying {:3f}".format(s))
ed_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=s,  # every 5 epochs
    decay_rate=0.1,
    staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=ed_scheduler)
model = keras.applications.resnet50.ResNet50(weights="imagenet")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


class PrintOnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print(model.optimizer.iterations.numpy(),
              model.optimizer._decayed_lr(tf.float32).numpy())


earlystopping_cb = keras.callbacks.EarlyStopping(patience=5)
tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())
callbacks = [earlystopping_cb, tensorboard_cb, PrintOnCallback()]
model.fit(
    train_data,
    validation_data=validate_data,
    epochs=15,
    verbose=1,
    callbacks=callbacks
)

results = model.evaluate(test_data)
print("test loss, test acc:", results)
