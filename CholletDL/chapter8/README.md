# How to setup the data:

## Get the data

- The kaggle setup doesn't work as I can't activate my account, I don't the the activation code my phone. No idea why. So I downloaded the data the microsoft website (google on cat's and dogs dataset).
- Run prepare cats_and_dogs.py to split up the data in train/validate/test datasets.

## Remove the invalid images
- Run find_corrupted_images.py
- Remove the listed images.

## Preprocess the images
- The rescaling happens in the model itself, so not seperate script required here.
- In the augmented case, the augmentations are also inside the model so not aditional script required to do the data augmentation.
