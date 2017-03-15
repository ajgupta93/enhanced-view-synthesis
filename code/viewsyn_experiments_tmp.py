import numpy as np
import viewsyn_model as model
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
import pdb
import shutil 
import random

# weights.69-543.30.hdf5
# weights.09-767.01.hdf5
# BN:
# weightsBN.14-636.57.hdf5
# weightsBN.29-562.16.hdf5
# weightsBN.19-0.94.hdf5
# weightsBN.29-0.95.hdf5

def load_test_data(current_chair_folder):
	img = []

	for filename in os.listdir(current_chair_folder):
		im = Image.open(current_chair_folder + filename).convert('RGB')
		img.append(np.asarray(im))
	return np.array(img)

def run_autoencoder(test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	hist = model.train_autoencoder(autoencoder)

	#test autoencoder
	autoencoder.load_weights('../model/weights.09-0.95.hdf5')
	model.test_autoencoder(autoencoder, test_images)

def get_autoencoder(weight_file):
	autoencoder = model.build_autoencoder()
	autoencoder.load_weights('../model/weights.09-0.95.hdf5')
	return autoencoder

if __name__ == '__main__':

	autoencoder = get_autoencoder('../model/weights.09-0.95.hdf5')
	test_folder = "../data/test/"
	for current_folder in os.listdir(test_folder):
		test_images = load_test_data(test_folder + current_folder + "/model_views/")
		model.test_autoencoder(autoencoder, test_images, current_folder + "/")

