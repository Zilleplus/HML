import numpy as np
import random
import os
import tensorflow.keras as keras  # type: ignore
from tensorflow.keras.utils import load_img, img_to_array, array_to_img  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

input_dir = "./dataset/images"
target_dir = "./dataset/annotations/trimaps"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)

target_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

img_size = (200, 200)
num_imgs = len(input_img_paths)

seed = 1337
random.Random(seed).shuffle(input_img_paths)
random.Random(seed).shuffle(target_paths)


def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img


input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = img_to_array(load_img(input_img_paths[i], target_size=img_size))
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

model = keras.models.load_model("oxford_segmentation.keras")

i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))
plt.show()

mask = model.predict(np.expand_dims(test_image, 0))[0]


def display_mask(pred):
    plt.figure()
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()


display_mask(mask)
