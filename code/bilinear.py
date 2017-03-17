import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential, Model
from keras.layers import *
from bilinear_layer import Bilinear
import tensorflow as tf
import pdb

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
	#pdb.set_trace()
	X = X + dx
	Y = Y + dy
	batch_size = 128
	#pdb.set_trace()
	O = tf.zeros([batch_size,224,224,3])
	tl = tf.slice(I, [0, 0, 0, 0], [batch_size, 222, 222, 3])
	tr = tf.slice(I, [0, 0, 2, 0], [batch_size, 222, 222, 3])
	bl = tf.slice(I, [0, 2, 0, 0], [batch_size, 222, 222, 3])
	br = tf.slice(I, [0, 2, 2, 0], [batch_size, 222, 222, 3])
	bilinear_output = (tl + tr + bl + br) / 4
	bilinear_output_with_padding = tf.pad(bilinear_output,[[0, 0], [1, 1], [1, 1], [0, 0]])
	# for batch_img in range(batch_size):
	# 	X_batch = tf.gather_nd(X,indices=[[batch_img]])
	# 	Y_batch = tf.gather_nd(Y,indices=[[batch_img]])
	# 	I_batch = tf.gather_nd(I,indices=[[batch_img]])
	# 	X_gather = tf.cast(tf.gather_nd(X_batch[0] ,indices=zip(range(1,h-1),range(1,w-1))),tf.int32) 
	# 	Y_gather = tf.cast(tf.gather_nd(Y_batch[0] ,indices=zip(range(1,h-1),range(1,w-1))),tf.int32)
	# 	#pdb.set_trace()
	# 	dim = int(X_gather.get_shape()[0])
	# 	avg_vals = []
	# 	x_indices = []
	# 	y_indices = []
	# 	for idx in range(dim):
	# 		x_idx = X_gather[idx]
	# 		y_idx = Y_gather[idx]
	# 		lt = tf.gather_nd(I_batch[0],indices=[[y_idx-1,x_idx-1]])
	# 		rt = tf.gather_nd(I_batch[0],indices=[[y_idx-1,x_idx+1]])
	# 		lb = tf.gather_nd(I_batch[0],indices=[[y_idx+1,x_idx-1]])
	# 		rb = tf.gather_nd(I_batch[0],indices=[[y_idx+1,x_idx+1]])
	# 		avg = (lt+rt+lb+rb)/4
	# 		avg_vals.append(avg)
	# 		x_indices.append(x_idx)
	# 		y_indices.append(y_idx)
	# 		pdb.set_trace()
	# 	sparse_tensor = tf.SparseTensor(indices=[y_indices,x_indices],values=avg_vals, shape=[224,224,3])
	# 	pdb.set_trace()

	return bilinear_output_with_padding
		
	#for x_idx, y_idx in zip(X_gather,Y_gather):

	# for y in range(1, h - 1):
	# 	for x in range(1, w - 1):
	# 		index = [y, x]
	# 		pdb.set_trace()
	# 		x_ = tf.gather_nd(X, tf.stack(index), -1)
	# 		val = getNeighbors(I, X_[y][x], Y_[y][x])
	# 		O[y][x][0:3] = val[0:3
	# 		#print(np.shape(val))
	
	

	
	# return O
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

	pdb.set_trace()

	img = image.load_img('../data/chair_test.png', target_size = (224, 224))
	img = image.img_to_array(img)
	img = np.asarray(img, dtype = np.int32)
	
	dx = np.zeros((224, 224))
	dy = np.zeros((224, 224))

	for i in range(150, 200):
		for j in range(150, 200):
			dy[i][j] = 5

	fp = np.concatenate((img, dx, dy), axis = 2)
	print fp.shape 
	pdb.set_trace()
	
	out = model.predict(fp)



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

