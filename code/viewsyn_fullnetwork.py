import keras
from keras.layers.convolutional import *
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.callbacks import *
from bilinear_layer import Bilinear
import utility as util
import pdb

def get_optimizer(name = 'adagrad', l_rate = 0.0001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	eps = 1e-8
	
	adam = Adam(lr = l_rate, beta_1 = b_1, beta_2 = b_2, epsilon = eps, decay = dec)
	sgd = SGD(lr = l_rate, momentum = mom, decay = dec, nesterov = True)
	rmsp = RMSprop(lr = l_rate, rho = rh, epsilon = eps, decay = dec)
	adagrad = Adagrad(lr = l_rate, epsilon = eps, decay = dec)
	
	optimizers = {'adam': adam, 'sgd':sgd, 'rmsp': rmsp, 'adagrad': adagrad}

	return optimizers[name]

def build_image_encoder():
	#define network architecture for image encoder
	model = Sequential()

	#6 convoltuion layers
	model.add(Convolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu',
			input_shape=(224, 224, 3)))
	model.add(Convolution2D(32, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))

	#Flatten 
	model.add(Flatten())

	#2 fully connected layers
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(p=0))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(p=0))

	return model

def build_viewpoint_encoder():
	#define network architecture for viewpoint transformation encoder
	model =  Sequential()

	#2 fully connected layers
	model.add(Dense(128, input_dim=19, activation='relu'))
	model.add(Dense(256, activation='relu'))

	return model
	

def build_common_decoder():

	#define network architecture for decoder
	model =  Sequential()

	#2 fully connected layers
	model.add(Dense(4096, input_dim=4352, activation='relu')) #4096+256
	model.add(Dense(4096, activation='relu'))
	
	#reshape to 2D
	model.add(Reshape((8, 8, 64)))
	
	#5 upconv layers
	model.add(Deconvolution2D(256, 3, 3, (None, 15, 15,256), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(128, 3, 3, (None, 29, 29, 128), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(64, 3, 3, (None, 57, 57, 64), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(32, 3, 3, (None, 113, 113, 32),border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(16, 3, 3, (None, 225, 225, 16), border_mode='same', subsample = (2,2), activation = 'relu'))
	
	return model

def output_layer_decoder(model, n_channel):
	
	#output layer, add 3 RGB channels for reconstructed output view
	model.add(Deconvolution2D(n_channel, 3, 3, (None, 225, 225, n_channel), border_mode='same', subsample = (1,1), activation = 'relu'))

	#add a resize layer to resize (225, 225) to (224, 224)
	model.add(Reshape((225*225,n_channel)))
	model.add(Lambda(lambda x: x[:,:50176,])) # throw away some
	model.add(Reshape((224,224,n_channel)))

	
	#print model.summary()
	return model


def build_full_network():
	image_encoder = build_image_encoder()
	view_encoder = build_viewpoint_encoder()

	decoder = build_common_decoder()
	decoder = output_layer_decoder(decoder, 5) #5

	#add bilinear layer
	decoder.add(Bilinear())

	mask_decoder = build_common_decoder()
	mask_decoder = output_layer_decoder(mask_decoder, 1)

	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)

	main_output = decoder(image_view_out)
	mask_output = mask_decoder(image_view_out)

	encoder_decoder = Model(input=[image_input, view_input], output=[main_output, mask_output])


	opt = get_optimizer('adam')
	encoder_decoder.compile(optimizer=opt, metrics=['accuracy'],
		loss={'sequential_3': 'mean_squared_error', 'sequential_4': 'binary_crossentropy'},
              loss_weights={'sequential_3': 1.0, 'sequential_4': 0.1})

	print encoder_decoder.summary()

	return encoder_decoder

def train_full_network(network):

	# note that we are passing a list of Numpy arrays as training data
	# since the model has 2 inputs and 2 outputs
	#Callbacks
	hist = History()
	checkpoint = ModelCheckpoint('../model/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
	callbacks_list = [hist, checkpoint]

	
	#history = network.fit([input_images, view_transformation], [output_views, masked_views], batch_size=64, nb_epoch=100, verbose=1, callbacks=callbacks_list,
	#	validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

	train_data_dict, val_data_dict = util.generate_data_dictionary()

	history = network.fit_generator(util.generate_data_from_list(train_data_dict), samples_per_epoch=64, nb_epoch=100, verbose=1, callbacks=callbacks_list,
		 validation_data=util.generate_data_from_list(val_data_dict), nb_val_samples=16, class_weight=None, initial_epoch=0)

	print hist.history
	return hist



