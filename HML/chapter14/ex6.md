# Can you name the main innovations in Alexnet compared to LeNet-5? What about the main innovations in GoogleNet, ResNet, SeNet and Xception.

AlexNet:
- They used local response normalization (LRN) !
- It was deeper then usual.
- They applied 50% dropout to compensate for beeing to deep/big
- They performed data augmentation by sifting th training images by different offsets/flipping them.

LeNet-5:
- It was one of the earlierst deep learning networks that used convolutional networks.

GoogleNet:
- They introduced inception modules (lots of less parameters then AlexNet)
- Depth concatenation layer (stack the features maps for all four top convolutional layers)

ResNet:
- Used a residual network -> It can skip connections, speeding up learning.

SeNet:
- Used a "SE block" a small neural network that thinks almost exclusively in depth. It detects which features are most active together. ANd uses this information to recalibrate the feature maps.

Xception:
- Extreme Inception "Depth wise seeparable convolution layer"
