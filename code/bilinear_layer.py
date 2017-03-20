from keras import backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
import pdb

# input image I, displacement matrix dx and dy
def binsample(I, dx, dy):
	batch_size = tf.shape(I)[0]

	x = range(224)
	y = range(224)

	X, Y = tf.meshgrid(x, y)
	X, Y = tf.cast( X, tf.float32 ), tf.cast( Y, tf.float32 )
	
	X = (X + dx) % 224
	Y = (Y + dy) % 224
	
	X, Y = tf.cast( X, tf.int32 ), tf.cast( Y, tf.int32 )
	
	_, Z, _ = tf.meshgrid(y, tf.range(batch_size), x)

	tl = tf.slice(I, [0, 0, 0, 0], [-1, 222, 222, 3])
	tr = tf.slice(I, [0, 0, 2, 0], [-1, 222, 222, 3])
	bl = tf.slice(I, [0, 2, 0, 0], [-1, 222, 222, 3])
	br = tf.slice(I, [0, 2, 2, 0], [-1, 222, 222, 3])
	
	simple_bilinear_output = (tl + tr + bl + br) / 4
	simple_bilinear_output_with_padding = tf.pad(simple_bilinear_output,[[0, 0], [1, 1], [1, 1], [0, 0]])
	
	curr_idx = tf.stack([Z, Y, X], axis = 3)
	
	transformed_idx = tf.reshape(curr_idx, [-1, 3])
	
	transformed_bilinear_image = tf.gather_nd(I, transformed_idx)
	img = tf.reshape(transformed_bilinear_image, [-1, 224, 224, 3])
	

	return img

class Bilinear(Layer):
	def __init__(self, **kwargs):
		self.bilinear = binsample
		self.input_spec = [InputSpec(ndim=4)]

		super(Bilinear, self).__init__(**kwargs)
		
	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]

	def call(self, x, mask=None):
		x_unstacked = tf.unstack(x, 5, axis=3)
		img = tf.stack(x_unstacked[0:3], axis=3)
		dx = x_unstacked[3]
		dy = x_unstacked[4]
		
		return self.bilinear(img, dx, dy)

	def get_config(self):
		config = {'bilinear': 'Bilinear layer'}
		base_config = super(Bilinear, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], input_shape[2], 3)