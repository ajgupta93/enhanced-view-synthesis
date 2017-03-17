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
	model.add(Bilinear(input_shape=(224, 224, 5)))

	print model.summary()

	current_chair_folder = "../data/bilinear_test/"
	test_data = load_test_data(current_chair_folder)
	
	out = model.predict(test_data)
	pdb.set_trace()
	util.save_as_image("../data/bilinear_output/", out)




if __name__ == '__main__':

	test_bilinear_layer()

