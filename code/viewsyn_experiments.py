import numpy as np
import viewsyn_model as model
import os

def load_test_data(test_folder):
	# img = []
	
	# for current_folder in os.listdir(test_folder):
	# 	for files in os.listdir(test_folder):
	# 		im = Image.open(test_folder + "model_views/" + filename).convert('RGB')
	# 		img.append(np.asarray(im))
	
	# return np.array(img)
	return None

def run_autoencoder(test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	hist = model.train_autoencoder(autoencoder)

	#test autoencoder
	# autoencoder.load_weights('../model/weights.19-225.48.hdf5')
	# model.test_autoencoder(autoencoder, test_images)
	

if __name__ == '__main__':
	test_folder = "../data/test/"
	test_images = load_test_data(test_folder)

	run_autoencoder(test_images)
