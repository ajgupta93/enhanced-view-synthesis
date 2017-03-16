from keras import backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
import pdb

class Bilinear(Layer):
	def __init__(self, bilinear_f, **kwargs):
		self.bilinear = bilinear_f
		super(Bilinear, self).__init__(**kwargs)
 		self.input_spec = [InputSpec(ndim=4)]

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]

	def call(self, x, mask=None):
		#print "hello",self.input_spec[0].shape
		x_unstacked = tf.unstack(x, 5, axis=3)
		img = tf.stack(x_unstacked[0:3], axis=3)
		dx = x_unstacked[3]
		dy = x_unstacked[4]
		
		return self.bilinear(img, dx, dy)

	def get_config(self):
		config = {'bilinear': 'Bilinear layer'}
		base_config = super(Bilinear, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))