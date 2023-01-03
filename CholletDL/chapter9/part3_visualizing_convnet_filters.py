import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow.keras as keras  # type: ignore
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt  # type: ignore


model = keras.applications.xception.Xception(weights="imagenet", include_top=False)

for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)

layer_name = "block3_sepconv1"
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)


def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    # Avoid border artifacts!
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # Reduce the image of the filter to one value:
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)

    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image


img_width = 200
img_height = 200


def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10
    image = tf.random.uniform(
        minval=0.4, maxval=0.6, shape=(1, img_height, img_height, 3))

    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()


def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]  # avoid border artifacts

    return image


plt.axis("off") # type: ignore
pattern = generate_filter_pattern(filter_index=2);
img = deprocess_image(pattern)
plt.imshow(img) # type: ignore
plt.show()

all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(generate_filter_pattern(filter_index=filter_index))
    all_images.append(image)

margin = 5
n = 8
cropped_width = img_width - 25*2
cropped_height = img_height - 25*2
width = n*cropped_width + (n-1)*margin
height = n*cropped_height + (n-1)*margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        image = all_images[i+n + j]
        row_start = (cropped_width + margin)*i
        row_end = (cropped_width + margin)*i + cropped_width
        column_start = (cropped_height + margin)*j
        column_end = (cropped_height + margin)*j + cropped_width

        stitched_filters[
            row_start: row_end,
            column_start: column_end,
            :
        ] = image
keras.utils.save_img(f"filters_for_layer_{layer_name}.png", stitched_filters)