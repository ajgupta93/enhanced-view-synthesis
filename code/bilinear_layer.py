from keras import backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
import pdb
import constants as const

# input image I, displacement matrix dx and dy
def binsample(I, dx, dy, batch_size = const.batch_size):
	
	#pdb.set_trace()
	
	h = 224 #np.shape(I)[0]
	w = 224 #np.shape(I)[1]

	O = np.zeros((h, w, 3))

	x = range(224)
	y = range(224)
	X, Y = tf.meshgrid(x, y)
	X, Y = K.cast( X, "float32" ), K.cast( Y, "float32" )
	X = (X + dx) % 224
	Y = (Y + dy) % 224
	X, Y = K.cast( X, "int32" ), K.cast( Y, "int32" )

	transformed_list = []
	tl = tf.slice(I, [0, 0, 0, 0], [-1, 222, 222, 3])
	tr = tf.slice(I, [0, 0, 2, 0], [-1, 222, 222, 3])
	bl = tf.slice(I, [0, 2, 0, 0], [-1, 222, 222, 3])
	br = tf.slice(I, [0, 2, 2, 0], [-1, 222, 222, 3])
	stacked_images = K.stack([tl, tr, bl, br])
	simple_bilinear_output = K.sum(stacked_images,axis=0)/4
	simple_bilinear_output_with_padding = K.spatial_2d_padding(simple_bilinear_output,(1, 1))
	for current_index in range(batch_size):
		curr_idx = tf.stack([Y[current_index], X[current_index]], axis = 2)
		transformed_idx = tf.reshape(curr_idx, [-1, 2])
		transformed_bilinear_image = tf.gather_nd(simple_bilinear_output_with_padding[current_index], transformed_idx)
		img = K.reshape(transformed_bilinear_image, [224, 224, 3])
		transformed_list.append(img)

	transformed_tensor = K.stack(transformed_list)
		
	return transformed_tensor

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
		dx = x_unstacked[3]
		dy = x_unstacked[4]
		
		return self.bilinear(img, dx, dy)

	def get_config(self):
		config = {'bilinear': 'Bilinear layer'}
		base_config = super(Bilinear, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], input_shape[2], 3)
