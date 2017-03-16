import keras
from keras.layers.convolutional import *
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.callbacks import *

def get_optimizer(name = 'adagrad', l_rate = 0.0001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	eps = 1e-8
	
	adam = Adam(lr = l_rate, beta_1 = b_1, beta_2 = b_2, epsilon = eps, decay = dec)
	sgd = SGD(lr = l_rate, momentum = mom, decay = dec, nesterov = True)
	rmsp = RMSprop(lr = l_rate, rho = rh, epsilon = eps, decay = dec)
	adagrad = Adagrad(lr = l_rate, epsilon = eps, decay = dec)
	
	optimizers = {'adam': adam, 'sgd':sgd, 'rmsp': rmsp, 'adagrad': adagrad}

	return optimizers[name]

def build_image_encoder():
	#define network architecture
	model = Sequential()

	#6 convoltuion layers
	model.add(Convolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'
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

def build_decoder():
	#2 fully connected layers
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(4096, activation='relu'))
	
	#reshape to 2D
	model.add(Reshape((8, 8, 64)))

	#5 upconv layers
	model.add(Deconvolution2D(256, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(128, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(64, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(32, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	
	#output layer, add 3 RGB channels for reconstructed output view
	model.add(Deconvolution2D(5, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))

	#add a resize layer which resize (225,225) to (224,224)

def build_autoencoder()
	
	encoder = build_image_encoder()
	decoder = build_decoder()

	model = Model(input=encoder.input, output=decoder.output)

	#compile model
	opt = get_optimizer('adam')
	model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

	print model.summary()

	return model


def train_autoencoder(autoencoder, input_data):
	X = input_data
	Y = input_data

	#Callbacks
	hist = History()
	checkpoint = ModelCheckpoint('../model/weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=5)
	callbacks_list = [hist, checkpoint]

	dataArr = util.generate_data_array_for_autoencoder(dataPath='./../data/train/')
	
	#for d in util.generate_autoencoder_data_from_list(dataArr):
	#	pdb.set_trace()

	history = autoencoder.fit_generator(util.generate_autoencoder_data_from_list(dataArr[:int(math.ceil(.8*len(dataArr)))]), samples_per_epoch = 293844, nb_epoch = 300, verbose=1, callbacks=callbacks_list,
		 validation_data=util.generate_autoencoder_data_from_list(dataArr[int(math.ceil(.8*len(dataArr))):]), nb_val_samples=128, class_weight=None, initial_epoch=0)

	print hist.history
	return hist

def test_autoencoder(autoencoder, test_data, sub_folder):
	
	test_image = test_data
	# pdb.set_trace()
	output = autoencoder.predict(test_image)
	
	# util.show_image(output)
	util.save_as_image("../data/autoencoder_results/" + sub_folder, output)