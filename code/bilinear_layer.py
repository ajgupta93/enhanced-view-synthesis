from keras import backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
import pdb
import constants as const

# Get the corresponding corner image for each pixel.
def get_sub_image(current_image, X_indices, Y_indices):
	current_indices = tf.stack([Y_indices, X_indices], axis = 2)
	reshape_indices = tf.reshape(current_indices, [-1, 2])
	sub_image = tf.gather_nd(current_image, reshape_indices)

	return K.reshape(sub_image,[224, 224, 3])

def element_wise_multiply(M1, M2):
	return K.prod(K.stack([M1, M2]), axis = 0)

def stack_up(W):
	# For 3 channels.
	return tf.stack([W, W, W], axis = 2)

# Input -> Image I, Matrix X and Matrix Y from which the pixel is to be extracted.
# Output -> The final image obtained from appearance flow and bilinear sampling.
def binsample(I, X, Y, batch_size = const.batch_size):
	
	# Get four corners for pixel (x,y)
	floor_X = tf.floor(X)
	floor_Y = tf.floor(Y)
	ceil_X = floor_X + 1
	ceil_Y = floor_Y + 1
	
	# Compute the distance of (x,y) from the corners.
	W_X = X - floor_X
	W_Y = Y - floor_Y

	# Compute the individual weightage of the four corners.
	W_tl = element_wise_multiply((1 - W_X), (1 - W_Y))
	W_bl = element_wise_multiply((1 - W_X), W_Y)
	W_tr = element_wise_multiply(W_X, (1 - W_Y))
	W_br = element_wise_multiply(W_X, W_Y)

	# Padding zeros along the boundaries to ease computation.
	padded_I = K.spatial_2d_padding(I,(1, 1))

	# Changing to base 1 due to padding.
	floor_X = floor_X + 1
	floor_Y = floor_Y + 1
	ceil_X = ceil_X + 1
	ceil_Y = ceil_Y + 1

	# Checking boundary conditions for X-coordinate.
	floor_X = K.maximum(floor_X, 0)
	floor_X = K.minimum(floor_X, I.get_shape()[1].value)
	ceil_X = K.maximum(ceil_X, 0)
	ceil_X = K.minimum(ceil_X, I.get_shape()[1].value)
	
	# Checking boundary conditions for Y-coordinate.
	floor_Y = K.maximum(floor_Y, 0)
	floor_Y = K.minimum(floor_Y, I.get_shape()[2].value)
	ceil_Y = K.maximum(ceil_Y, 0)
	ceil_Y = K.minimum(ceil_Y, I.get_shape()[2].value)

	# Typecasting to int32
	floor_X = K.cast(floor_X, tf.int32)
	floor_Y = K.cast(floor_Y, tf.int32)
	ceil_X = K.cast(ceil_X, tf.int32)
	ceil_Y = K.cast(ceil_Y, tf.int32)
	
	weighted_image_list = []
	
	
	for current_index in range(batch_size):
		current_image = padded_I[current_index]
		# Calculate the corresponding images for the four possible corners for each pixel.
		image_tl = get_sub_image(current_image, floor_X[current_index], floor_Y[current_index])
		image_bl = get_sub_image(current_image, floor_X[current_index], ceil_Y[current_index])
		image_tr = get_sub_image(current_image, ceil_X[current_index], floor_Y[current_index])
		image_br = get_sub_image(current_image, ceil_X[current_index], ceil_Y[current_index])
		
		# Construct the weighted image for each pixel.
		weighted_image = element_wise_multiply(stack_up(W_tl[current_index]), image_tl) +\
						 element_wise_multiply(stack_up(W_bl[current_index]), image_bl) +\
						 element_wise_multiply(stack_up(W_tr[current_index]), image_tr) +\
						 element_wise_multiply(stack_up(W_br[current_index]), image_br)
		# pdb.set_trace()
		weighted_image_list.append(weighted_image)

	output_tensor = K.stack(weighted_image_list)
		
	return output_tensor

class Bilinear(Layer):
	def __init__(self, **kwargs):
		self.bilinear = binsample
		self.input_spec = [InputSpec(ndim=4)]

		super(Bilinear, self).__init__(**kwargs)
		
	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		#self.kernel = self.add_weight(shape=(1, 1),
		#	initializer='uniform', trainable=False)
	
	def call(self, x, mask=None):
		x_unstacked = tf.unstack(x, 5, axis=3)
		img = tf.stack(x_unstacked[0:3], axis=3)
		x = x_unstacked[3]
		y = x_unstacked[4]
		
		return self.bilinear(img, x, y)

	def get_config(self):
		config = {'bilinear': 'Bilinear layer'}
		base_config = super(Bilinear, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], input_shape[2], 3)
