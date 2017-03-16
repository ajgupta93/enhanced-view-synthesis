import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential, Model
from bilinear_layer import Bilinear
import tensorflow as tf
import pdb

# input image I, displacement matrix dx and dy
def binsample(I, dx, dy):
	sess = tf.Session()
	pdb.set_trace()
	I = I.eval(session=sess)
	dx = dx.eval(session=sess)
	dy = dy.eval(session=sess)

	h = np.shape(I)[0]
	w = np.shape(I)[1]

	O = np.zeros((h, w, 3))

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			val = getNeighbors(I, x + dx[y][x], y + dy[y][x])
			O[y][x][0:3] = val[0:3]
			#print(np.shape(val))
	
	x = range(224)
	y = range(224)

	X, Y = tf.meshgrid(x, y)

	
	return O
def getNeighbors(I, x, y):
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

