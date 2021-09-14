from tensorflow import keras
import os

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only=True)
earlystopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
# enable callback you want here:
callbacks = [earlystopping_cb]
#tensorboard_cb,
#checkpoint_cb,

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

learning_rate = 2e-3
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
optimizer = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=100,
          validation_split=0.2,
          verbose=1,
          callbacks=callbacks
          )
[loss, accuracy] = model.test_on_batch(x=x_test,
                                       y=y_test)
print("Loss={0}".format(loss))
print("Accuracy={0}".format(accuracy))

# Learning rate of 2e-3 gives accuracy of 95.4%
# using the ReLu activation function.
#
# using the sigmoid function (activation='sigmoid')
# with learning rate=2e-3 we get about 94.5% accuracy.
# Maybe I should play around with this learning rate,
# but it's very sloooooooooow to train that.
