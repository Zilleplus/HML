from pickletools import optimize
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

inputs = keras.Input(shape=(28, 28, 1))
# 32 filters of 3*3 parameters = 288 parameters
# Every filter also has a bias (32*1 parameter) -> 288 params + 32 params = 320 params
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
# 32 input filters and 64 outputs filters, 3*3 window + bias per filter
# = 32*64*(3*3) + 64 = 18496
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(units=10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# output
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
#                                                                  
#  conv2d_6 (Conv2D)           (None, 26, 26, 32)        320       
#                                                                  
#  max_pooling2d_4 (MaxPooling  (None, 13, 13, 32)       0         
#  2D)                                                             
#                                                                  
#  conv2d_7 (Conv2D)           (None, 11, 11, 64)        18496     
#                                                                  
#  max_pooling2d_5 (MaxPooling  (None, 5, 5, 64)         0         
#  2D)                                                             
#                                                                  
#  conv2d_8 (Conv2D)           (None, 3, 3, 128)         73856     
#                                                                  
#  flatten_2 (Flatten)         (None, 1152)              0         
#                                                                  
#  dense_2 (Dense)             (None, 10)                11530     
#                                                                  
# =================================================================
# Total params: 104,202
# Trainable params: 104,202
# Non-trainable params: 0

(train_images , train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)