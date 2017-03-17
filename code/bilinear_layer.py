from keras import backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
import pdb

class Bilinear(Layer):
	def __init__(self, bilinear_f, **kwargs):
		self.bilinear = bilinear_f
		self.input_spec = [InputSpec(ndim=4)]

		super(Bilinear, self).__init__(**kwargs)
		
	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]
		self.input_dim = input_dim

	def call(self, x, mask=None):
		#print "hello",self.input_spec[0].shape
		print ">:(" , x[0][:,:,1][0][0]
		x_unstacked = tf.unstack(x, 5, axis=3)
		img = tf.stack(x_unstacked[0:3], axis=3)
		dx = x_unstacked[3]
		dy = x_unstacked[4]
		
		return self.bilinear(img, dx, dy)

	def get_config(self):
		config = {'bilinear': 'Bilinear layer'}
		base_config = super(Bilinear, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	# def _fix_unknown_dimension(self, input_shape, output_shape):
	#   output_shape = list(output_shape)

	#   msg = 'total size of new array must be unchanged'

	#   known, unknown = 1, None
	#   for index, dim in enumerate(output_shape):
	#       if dim < 0:
	#           if unknown is None:
	#               unknown = index
	#           else:
	#               raise ValueError('Can only specify one unknown dimension.')
	#       else:
	#           known *= dim

	#   original = np.prod(input_shape, dtype=int)
	#   if unknown is not None:
	#       if known == 0 or original % known != 0:
	#           raise ValueError(msg)
	#       output_shape[unknown] = original // known
	#   elif original != known:
	#       raise ValueError(msg)

	#   return tuple(output_shape)

	# def get_output_shape_for(self, input_shape):
	# 	# pdb.set_trace() 
	# 	return [input_shape[1:]] * 4
	# 	# return (input_shape[0],) + self._fix_unknown_dimension(input_shape[1:], self.target_shape)


	# def compute_mask(self, inputs, mask=None):
	# 	# pdb.set_trace()
	# 	return [(224,224,5)] * 4