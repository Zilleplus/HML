# Chapter14 Deep Computer Vision using Convolutional neural networks

## Convolutional layers
- Neurons are only connected to neurons in a perceptive field, this reduces the number of to learn parameters.
- Zero padding adds zero's to the edges so the next layer has the same number of neurons (with stride=1).
- The stride(vertical/horizontal) defines the shift from one perceptive field to the next one.

## Filters (Convolutional layers)
- Filter + neurons = feature map
- Multiple filters can be connected to the same neural network.

$ z_{i,j,b} = b_k + \sum_{u=0}^{f_h-1} \sum_{v=0}^{f_w -1} \sum_{k=0}^{f_{n'}} x_{i',j',k'} w_{u, v, k', k}$
with
$i' = i s_h + u$: loop variable height in preception field
$j' = j s_w + v$: loop variable width in preception field


- z_{i, j, k}
- s_h/s_w: horizontal/vertical stride
- f_w/f_h: width/height reception field
- f_n: the number of feature maps

- w_{u,v,k',k}: weight between any feature map k of the layer l and its input located at row u, column v and feature map k'
- x_{i', j', k'}: output of the neuron of previous layer k'

## Tensorflow implementation
input = [batch_size, height, width, channels]

padding = 
| "SAME"  : zero padding
| "VALID" : no padding

## Memory requirements
(featuremaps size + neurons)* num_of_feature_maps

## Pooling layers
Pooling layers reduce the size of the image without using parameters.
some examples of pooling layers
- Down sampling
- Averageing 
- max pooling

Advantages:
- less parameters
- less memory
- invariant to small translation/rotation changes
Disadvantages:
- Destructive: Some information is lost, might cause trouble for certain applications

note: Pooling can also be done depth wise(pooling together filters), but this less common.

## Data Augmentation
Generate more images by modifying exising ones. (rotate/translate/scale/add noise/...) This is a kinda of regularization and reduces overfitting.

## Local Response Normalization
A highly activated neuron activates other neurons on the same spot but different feature map. This pushes the feature maps to specialize more, and so increases generalization.

b_i = a_t (k + \alpha \sum_{j_low}^{j_high}a_j^2)
with:
j_high = min(i+r/2, f_n-1)
j_low = max(0, i-r/2)

b_i: is the normalized output of the neural located in feature map, at row u, collumn v. (LRN only work depth wise)
a_i: is the activation of that neuron after the ReLU step, but before normalization.
k, \alpha, \beta, r: hyper parameters k=bias, r=depth radius
f_n: the number of feature maps

## LeNet5
A combination of conv2d layers, and average pooling. The output layer is an euclidian distance, wich is rather unique.

## AlexNet
Builds upon LeNet, but is much deeper and uses max pooling. I uses 50% dropout, and data augmentation (rotation, scaling, flipping horizontally, change lighting).

## GoogLeNet
Google net is the first network to use inception modules, the inception models allow it to be much deeper then AlexNet. But have fewer parameters then AlexNet.

An inception module exists out of several CNN's with stride=1 and padding="same". The CNN's with kernels larger then 1, have an extra layer with a kernel=1 before or after them. Finally all the different CNN's are connected together using a "Depth Concat" (tf.Concat(...)). Concatting together the different feature maps.

1*1 layer + n*n layer combination has 3 advantages:
1. Pattern trought depth.
2. Bottleneck layers : They reduce the number of feature map -> reduce the dimensionality.
3. Combinations like [1*1,3*3] can capture very complex patterns.

## VGGNET
Not much special things about VGGNET, just a regular setup using repeated 2/3 CNN's followed by a pooling layer.

## ResNet: Residual network
This is a very deep network, too train it (and not have vanishing gradient) residual units are used. The output of a block with a few convolutional layers is connected to the input and output of the next block. Allowing the network to skip a block. This allows the training to take place much quicker in higher blocks. These connections are called skip connections or shortcut connections.


## Xception
A depth wise separable convolution layer (or short: seperable convolution layer), is a layer that only works on 1 channel. So only spacial information is used. By combining a seperable convolution layer with a 1*1 convolution layer(one that only does depth, and no special). We get something very similar to the inception layers of GoogleNet, but in practice it behaves a bit better.

## SENet(inception networks + ResNet)
SENet uses inception/residual units just like ResNet does, but it adds SEBlocks too it. 

An SeBlock Exists out of 3 parts
1. Global AVG: get 1 number from each channel/feature
2. Dense layer(few neurons): low number of neurons, this is where the squeeze happens. Only a limited number of features are allowed.
3. Dense layer(lots of neurons): expand again to get a number per input channel. The input channels are then multiplied by there corresponding number.

## Object detection
Classification and localization of objects. 

### Common approach
Slide a CNN classifier over the image, then use the non-max suppresion to remove duplicate boxes.

1. Use "non-flower" class to get rid of some of the boxes. (use sigmoid activation function and binary cross entropy loss function)
2. Find bounding boxes with highest score and eliminate overlapping boxes. (eg. more then 60% IoU)
3. Repeat from step 2 untill no more boxes are removed.

### Fully convolutional network
Replace the op dense later by an CNN. The output then contains feature maps that represent the sliding window, but are much much cheaper to evaluate.

example:
- Original classifier: CNN (7*7 with 100 feature maps) + dense layer of 200 neurons
- Fully convolutional network: CNN (7*7 with 100 feature maps) + CNN of 200 features of (1*1)

Because of the 1*1 feature maps, the same weights can be used in the FCNN as in the original dense layer. When the input image size is doubled, in width and length. The output feature map becomes 2*2. Representing the sliding window over the 4 quadrants.

### You only only look once

Yolov3:
- Outputs 5 bouding boxes for each grid ceel 
- 20 class probabilities
- 5 objectiveness scores
- Uses relative coordinates, (0,0) is top left, (1,1) is bottom right
- Before training the network it uses K-means to find bouding boxes (these are called anchor boxes/bounding boxes)
- !! How the scaling exactly happens is somewhat vague in the book, but on page 490 3 papers are referenced, and recommonded to read. I assume a detailed explanation can be found inside them.

### Mean average precision
1. ROC curve contains precision vs recall.
2. What is the best precision with a minimum recall of ... % -> do this for different value of the recall. This is called "average precision"
3. Take the average of the "average precision" of different classes, this is call the "Mean average precision"

## Segment Segmentation
Classify every pixel of the image inside a class.

- The regular CNN's lose resolution on every layer, so upsampling is required
- The "Transposed convolution layer" inserts zero rows and collumns to get the same size. It then applies the convolution step. (tf.keras.Conv2DTransposed)

Some other keras layers are:
- Conv1d
- Conv3d
- dilation rate (hyperparameter of CNN's): this inserts zeros in rows/collumns to increase the size of the perception field without any computational cost to it.
- dethwise_conv2d: applies all filters seperately to all channels, lots of output feature maps!

Additional improvements:
- skipping connections: Upsample a layer, and add a lower layer (one that has the same resolution as this layers upsampled) to it. 
- super-resolution: Upsample beyond the original image
- adversarial learning
- single shot learning
- ... and many more, seems like 1 chapter is not even close to enough
