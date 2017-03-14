import numpy as np
import viewsyn_model as model
import viewsyn_fullnetwork as fnetwork
import utility as util

def load_data():
	train = np.load('../data/train_images_autoencoder.npy')
	test = np.load('../data/test_images_autoencoder.npy')

	return train, test

def run_autoencoder(train_images, test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	hist = model.train_autoencoder(autoencoder, train_images)

	#test autoencoder
	autoencoder.load_weights('../model/weights.19-225.48.hdf5')
	model.test_autoencoder(autoencoder, test_images)
	

def run_full_network():
	full_network = fnetwork.build_full_network()


if __name__ == '__main__':
	train_images, test_images = load_data()

	#run_autoencoder(train_images, test_images)

	#run_full_network()

	util.generate_data_dictionary()
