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

def load_test_data():
	im = Image.open("../data/chairs/ec7076f7f37a6e124e234a6f5a66d6d3/model_views/55_30.png").convert('RGB')
	img = []
	img.append(np.asarray(im))
	return np.array([img])

def run_autoencoder(train_images, test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	# hist = model.train_autoencoder(autoencoder)

	#test autoencoder
	autoencoder.load_weights('../model/weightsBN.94-0.93.hdf5')
	model.test_autoencoder(autoencoder, test_images)
	
if __name__ == '__main__':

	test_images = load_test_data()
	print test_images.shape
	run_autoencoder(test_images)
