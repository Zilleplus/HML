# Consider a DNN composed of three convulutional layers, each with 3 x 3 kernels, a stride of 2, and "same" padding. Te lowest layer outputs 100 feature maps, the middle one outputs 200, and the top one outputs 400. The input emages are RBG images of 200*300 pixels.

What is the total number of parameters in the CNN? 

Top:

3*3 kernel with 3 colors and 1 bias term
3*3*3 + 1 = 28
100 feature maps 
100*28 = 2800

Middle:

3*3 kernel 
100 input's (feature maps top layer)
3*3*100 = 900

add 1 bias term per output
1 bias

200 outputs: 200*901 = 18200

Lower:

200 input feature maps
3*3 kernel

200 * 3*3 = 1800
1 bias term

400 output feature maps
1801 * 400 = 720 400

Total:
903.400
If we are using 32-bit floats, at least how much RAM will this network require when making a prediction for a single instance? 

Stride = 2 -> filters shrink by scale of 4 every layer

filter_size_first_layer = 300*200/(2*2) = 150*100 = 15000
filter_size_second_layer = filter_size_first_layer / (2*2) = 150*100/4 = 75*50 = 3750
filter_size_third_layer = filter_size_second_layer / (2*2) = 38*25 = 950

first_layer_size =  100*15000 = 1 500 000 floats => 6 MByte
second_layer_size = 200*3750 = 750 000 floats => 3MByte
third_layer_size = 400*950 = 380 000 floats => 1.52MByte

We only need to keep track of 1 layer at the time.
903400 floats = 3.6MByte

so just after calculating the second layer we hit a peak memory usuage just before releasing the memory of the first layer. The peak is about 6+3 = 9MByte filters and 3.6MByte paramters.
This brings the total to 12.6MByte


what about when training on a mini-batch of 50 images?

50*the output layer = 50* (6+3+1.5)MByte = 525MByte
input images = (50 images)*(300*200 resolution)*(3 colors) floats = 4*50*300*200*3 bytes = 36MByte

The cost of the gradient is neglectable.

This brings the total cost to about 36+525MByte = 561MByte
