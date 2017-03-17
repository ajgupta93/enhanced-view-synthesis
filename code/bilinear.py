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


# input image I, displacement matrix dx and dy
def binsample(I, dx, dy):
	
	#pdb.set_trace()
	
	h = 224 #np.shape(I)[0]
	w = 224 #np.shape(I)[1]
	batch_size = I.get_shape().as_list()[0]

	O = np.zeros((h, w, 3))

	x = range(224)
	y = range(224)
	z = None
	#pdb.set_trace()
	if batch_size is None:
		z = 1
	else:
		z = range(batch_size)
	X, Y, Z = tf.meshgrid(y, z, x)
	X, Y, Z = tf.cast( X, tf.float32 ), tf.cast( Y, tf.float32 ), tf.cast( Z, tf.float32 )
	#pdb.set_trace()
	X = (X + dx) % 224
	Y = (Y + dy) % 224
	x, y, z = tf.cast( X, tf.int32 ), tf.cast( Y, tf.int32 ), tf.cast( Z, tf.int32 )


	# tl = tf.slice(I, [0, 0, 0, 0], [batch_size, 222, 222, 3])
	# tr = tf.slice(I, [0, 0, 2, 0], [batch_size, 222, 222, 3])
	# bl = tf.slice(I, [0, 2, 0, 0], [batch_size, 222, 222, 3])
	# br = tf.slice(I, [0, 2, 2, 0], [batch_size, 222, 222, 3])
	# simple_bilinear_output = (tl + tr + bl + br) / 4
	# simple_bilinear_output_with_padding = tf.pad(simple_bilinear_output,[[0, 0], [1, 1], [1, 1], [0, 0]])
	transformed_list = []
	# pdb.set_trace()
	#for current_index in range(I.get_shape().as_list()[0]):
	# pdb.set_trace()
	#current_image = I[current_index]
	tl = tf.slice(I, [0, 0, 0, 0], [-1, 222, 222, 3])
	tr = tf.slice(I, [0, 0, 2, 0], [-1, 222, 222, 3])
	bl = tf.slice(I, [0, 2, 0, 0], [-1, 222, 222, 3])
	br = tf.slice(I, [0, 2, 2, 0], [-1, 222, 222, 3])
	# pdb.set_trace()
	simple_bilinear_output = (tl + tr + bl + br) / 4
	simple_bilinear_output_with_padding = tf.pad(simple_bilinear_output,[[0,0], [1, 1], [1, 1], [0, 0]])
	# pdb.set_trace()
	x = tf.reshape(x,[-1])
	y = tf.reshape(y,[-1])
	z = tf.reshape(z,[-1])
	#pdb.set_trace()

	curr_idx = tf.stack([z, y, x], axis = 1)
	#transformed_idx = tf.reshape(curr_idx, [-1, 2])
	J = tf.reshape(I,[-1,3])
	transformed_bilinear_image = tf.gather_nd(J, curr_idx)
	img = tf.reshape(transformed_bilinear_image, [-1, 224, 224, 3])
	#transformed_list.append(img)
	#transformed_tensor = tf.stack(transformed_list)
	#return transformed_list
	return img

def getNeighbors(I, x, y):
	pdb.set_trace()
	
	
	y = int(y)
	x = int(x)

	lt = I[y - 1][x - 1]
	rt = I[y - 1][x + 1]
	lb = I[y + 1][x - 1]
	rb = I[y + 1][x + 1]
	#print(lt.size())

	return 	((lt + rt)/2 + (lb + rb)/2)/2


def test_bilinear_layer():
	model = Sequential()
	model.add(Bilinear(binsample, input_shape=(224, 224, 5)))

	print model.summary()

	img = image.load_img('../data/chair_test.png', target_size = (224, 224))
	img = image.img_to_array(img)
	img = np.asarray(img, dtype = np.int32)
	
	dx = np.zeros((224, 224,1))
	dy = np.zeros((224, 224,1))

	for i in range(224):
		for j in range(224):
			dy[i][j][0] = 50

	fp = np.concatenate((img, dx, dy), axis = 2)
	#fp = np.array([fp, fp, fp])

	img = image.load_img('../data/chair_test_2.png', target_size = (224, 224))
	img = image.img_to_array(img)
	img = np.asarray(img, dtype = np.int32)
	
	dx = np.zeros((224, 224,1))
	dy = np.zeros((224, 224,1))

	for i in range(224):
		for j in range(224):
			dy[i][j][0] = 25

	fp = np.array([fp,np.concatenate((img, dx, dy), axis = 2)])

	print fp.shape 

	pdb.set_trace()
	out = model.predict(fp)
	util.save_as_image("../data/", out)




if __name__ == '__main__':
	# img = image.load_img('../data/chair_test.png', target_size = (224, 224))
	# img = image.img_to_array(img)
	# img = np.asarray(img, dtype = np.int32)
	
	# dx = np.zeros((224, 224))
	# dy = np.zeros((224, 224))

	# for i in range(150, 200):
	# 	for j in range(150, 200):
	# 		dy[i][j] = 5

	# #print(I[1][1])
	# out = binsample(img, dx, dy)

	# plt.imshow(out.astype(np.uint8))
	# plt.show()

	test_bilinear_layer()

