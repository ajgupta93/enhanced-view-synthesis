import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential, Model
from keras.layers import *
from bilinear_layer import Bilinear
import tensorflow as tf
import pdb
import utility as util
import os


# input image I, displacement matrix dx and dy
def binsample(I, dx, dy):
	
	#pdb.set_trace()
	
	h = 224 #np.shape(I)[0]
	w = 224 #np.shape(I)[1]

	O = np.zeros((h, w, 3))

	x = range(224)
	y = range(224)
	X, Y = tf.meshgrid(x, y)
	X, Y = tf.cast( X, tf.float32 ), tf.cast( Y, tf.float32 )
	X = (X + dx) % 224
	Y = (Y + dy) % 224
	X, Y = tf.cast( X, tf.int32 ), tf.cast( Y, tf.int32 )
	# cur_idx = tf.stack([X, Y], axis = 2)
	# new_idx = tf.reshape(cur_idx, [-1, 2])
	# lin_img = tf.gather_nd(I, new_idx)
	# img = tf.reshape(lin_img, [224, 224, 3])
	# pdb.set_trace()
	batch_size = 1

	# tl = tf.slice(I, [0, 0, 0, 0], [batch_size, 222, 222, 3])
	# tr = tf.slice(I, [0, 0, 2, 0], [batch_size, 222, 222, 3])
	# bl = tf.slice(I, [0, 2, 0, 0], [batch_size, 222, 222, 3])
	# br = tf.slice(I, [0, 2, 2, 0], [batch_size, 222, 222, 3])
	# simple_bilinear_output = (tl + tr + bl + br) / 4
	# simple_bilinear_output_with_padding = tf.pad(simple_bilinear_output,[[0, 0], [1, 1], [1, 1], [0, 0]])
	transformed_list = []
	for current_index in range(batch_size):
		# pdb.set_trace()
		current_image = I[current_index]
		tl = tf.slice(current_image, [0, 0, 0], [222, 222, 3])
		tr = tf.slice(current_image, [0, 2, 0], [222, 222, 3])
		bl = tf.slice(current_image, [2, 0, 0], [222, 222, 3])
		br = tf.slice(current_image, [2, 2, 0], [222, 222, 3])
		# pdb.set_trace()
		simple_bilinear_output = (tl + tr + bl + br) / 4
		simple_bilinear_output_with_padding = tf.pad(simple_bilinear_output,[[1, 1], [1, 1], [0, 0]])
		# pdb.set_trace()
		curr_idx = tf.stack([Y[current_index], X[current_index]], axis = 2)
		transformed_idx = tf.reshape(curr_idx, [-1, 2])
		transformed_bilinear_image = tf.gather_nd(current_image, transformed_idx)
		img = tf.reshape(transformed_bilinear_image, [1, 224, 224, 3])
		transformed_list.append(img)
	transformed_tensor = tf.stack(transformed_list)
	return transformed_list

def load_test_data(current_chair_folder):
	img = []

	for filename in os.listdir(current_chair_folder):
		if filename == ".DS_Store": continue
		im = image.img_to_array(image.load_img((current_chair_folder + filename)))
		# pdb.set_trace()
		dx = np.zeros((224, 224,1))
		dy = np.zeros((224, 224,1))
		for i in range(224):
			for j in range(224):
				dy[i][j][0] = 50
		fp = np.concatenate((im, dx, dy), axis = 2)
		img.append(np.asarray(fp))
	return np.array(img)


def test_bilinear_layer():
	model = Sequential()
	model.add(Bilinear(binsample, input_shape=(224, 224, 5)))

	print model.summary()

	current_chair_folder = "../data/bilinear_test/"
	test_data = load_test_data(current_chair_folder)
	pdb.set_trace()
	# util.save_as_image("../data/", [img])
	out = model.predict(test_data)
	pdb.set_trace()
	util.save_as_image("../data/bilinear_output/", out)




if __name__ == '__main__':

	test_bilinear_layer()

