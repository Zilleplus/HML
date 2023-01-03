import numpy as np
import tensorflow as tf # type: ignore
import tensorflow.keras as keras  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.cm as cm

model = keras.applications.xception.Xception(weights="imagenet")

img_path = keras.utils.get_file(
    fname="elephant.jpg", origin="https://img-datasets.s3.amazonaws.com/elephant.jpg"
)


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = keras.applications.xception.preprocess_input(array)
    return array


img_array = get_img_array(img_path, target_size=(299, 299))

preds = model.predict(img_array)
top_3_preds = keras.applications.xception.decode_predictions(preds, top=3)[0]

# Or we can do it manually with np.argmax (tf.math.argmax doesn't seem to work not sure why).
highest_prob_preds = np.argmax(preds[0])

last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = ["avg_pool", "predictions"]

last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)

    # Ensures that `tensor` is being traced by this tape.
    tape.watch(last_conv_layer_output)

    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(target=top_class_channel, sources=last_conv_layer_output)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

heatmap = np.mean(last_conv_layer_output, axis=-1)

img = keras.utils.load_image(img_path)
img = keras.utils.img_to_array(img)
heatmap = np.uint8(255 * heatmap)

# cm = colormap
jet = cm.get_cmap("jet")
