import numpy as np
import viewsyn_model as model
import viewsyn_fullnetwork as fnetwork
import utility as util
import h5py
import pdb

def load_test_data():
	#TODO
	return None

def run_autoencoder(test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	hist = model.train_autoencoder(autoencoder)

	#test autoencoder
	#autoencoder.load_weights('../model/weights.19-225.48.hdf5')
	#model.test_autoencoder(autoencoder, test_images)
	

def run_full_network():
	weights_path = '../model/weights.29-0.95.hdf5'
	full_network = fnetwork.build_full_network()
	# fnetwork.load_autoenocoder_model_weights(full_network, weights_path)

	fnetwork.train_full_network(full_network)

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
				print "exception!"
				pdb.set_trace()
		except:
			pdb.set_trace()






if __name__ == '__main__':
	# train_images, test_images = load_data()

	#run_autoencoder(train_images, test_images)

	run_full_network()
	# test_load_weights()

	#util.generate_data_dictionary()
