from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import *
from bilinear_layer import Bilinear
import numpy as np
import tensorflow as tf
import utility as util
import h5py, pdb, os
import viewsyn_model as model
import viewsyn_fullnetwork as fnetwork
import random

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

def load_test_data_replication(current_chair_folder):
	img = []
	view_transform = []

	filename = current_chair_folder+'input.png'

	for i in range(19):
		im = image.img_to_array(image.load_img(filename))
		img.append(np.asarray(im))
		
		vp = np.zeros(19)
		#index = random.randint(0, 18)
		vp[i] = 1

		view_transform.append(vp)
	
	return [np.array(img), np.array(view_transform)]

def test_load_weights():
	weights_path = '../model/weights.29-0.95.hdf5'
	f = h5py.File(weights_path)

	# Create Full Network with autoencoder weight initialization
	full_network = fnetwork.build_full_network()
	fnetwork.load_autoenocoder_model_weights(full_network, weights_path)
	
	# Create autoencoder network for test purpose only
	autoencoder = model.build_autoencoder()
	autoencoder.load_weights(weights_path)

	layers = full_network.layers
	
	image_encoder_network = layers[2].layers
	image_decoder_network = layers[5].layers
	
	# Subset of layer that acts as autoencoder in full network
	combined_network = np.concatenate((image_encoder_network, image_decoder_network))

	# Layer names in both the network
	layer_auto_n = ['convolution2d_7', 'convolution2d_8', 'convolution2d_9', 'convolution2d_10', 'convolution2d_11', 'convolution2d_12', 'flatten_2', 'dense_9', 'dropout_3', 'dense_10', 'dropout_4', 'dense_11', 'dense_12', 'reshape_7', 'deconvolution2d_13', 'deconvolution2d_14', 'deconvolution2d_15', 'deconvolution2d_16', 'deconvolution2d_17', 'deconvolution2d_18', 'reshape_8', 'lambda_3', 'reshape_9']
	layer_full_n = [x.name for x in combined_network]
	
	# Load weights layer by layer and compute the difference. Exception occurs for flatten, dropout, reshape layers.
	for i in range(len(layer_full_n)):
		try:
			layer_name_full = layer_full_n[i]
			layer_name_auto = layer_auto_n[i]
			
			layer_auto = autoencoder.get_layer(layer_name_auto)
			layer_full = combined_network[i]
			
			w_auto = layer_auto.get_weights()[0]
			w_full = layer_full.get_weights()[0]
			try:
				diff = (w_auto - w_full) ** 2
				print np.sum(diff)
			except:
				print "Inner Exception, for layer:", layer_full_n[i]
				pdb.set_trace()
		except:
			print "Exception for layer:", layer_full_n[i]
			pdb.set_trace()

def test_bilinear_layer():
	model = Sequential()
	model.add(Bilinear(input_shape=(224, 224, 5)))

	print model.summary()

	current_chair_folder = "../data/chairs/eb3029393f6e60713ae92e362c52d19d/model_views/"
	test_data = load_test_data(current_chair_folder)
	
	out = model.predict(test_data)
	pdb.set_trace()
	util.save_as_image("../data/chairs/eb3029393f6e60713ae92e362c52d19d/", out)

def test_full_network():
	weights_path = '../model/weights.39-618.91.hdf5'
	
	full_network = fnetwork.build_full_network()
	full_network.load_weights(weights_path)

	current_chair_folder = "../data/chairs/eb3029393f6e60713ae92e362c52d19d/model_views/"
	test_data = load_test_data(current_chair_folder)
	
	out = full_network.predict(test_data)

	util.save_as_image("../data/chairs/eb3029393f6e60713ae92e362c52d19d/test_fullNet/", out)

def test_replication_network():
	weights_path = '../model/weights.09-10.49.hdf5'
	
	full_network = fnetwork.build_full_network()
	full_network.load_weights(weights_path)

	current_chair_folder = "../data/chairs/eb3029393f6e60713ae92e362c52d19d/test_replication/input/"
	test_data = load_test_data_replication(current_chair_folder)
	
	out = full_network.predict(test_data)
	util.save_as_image("../data/chairs/eb3029393f6e60713ae92e362c52d19d/test_replication/", out[0])

	
	layer_name = 'sequential_3'
	#pdb.set_trace()
	intermediate_layer_model = Model(input=full_network.input, output=full_network.get_layer(layer_name).get_output_at(1))
	intermediate_output = intermediate_layer_model.predict(test_data)
	#pdb.set_trace()
	
	
if __name__ == '__main__':
	#Remember to set batch_size accordingly.
	#test_bilinear_layer()
	#test_load_weights()
	test_replication_network()
