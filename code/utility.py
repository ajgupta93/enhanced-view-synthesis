from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from PIL import Image
import os
import pdb
import shutil 

def save_as_image(images):
	for i in range(0, len(images)):
		filename = filepath+str(i)+".png"
		imsave(filename, images[i])

def show_image(image):
	plt.imshow(np.squeeze(image))
	plt.show()

def img_mask_gen(imgpath):
	im = Image.open(imgpath).convert('L').point(lambda x: 0 if x<=0 or x>=250 else 255,'1')
	return im	

def get_azimuth_transformation(in_path, out_path):
	in_f = in_path.split('/')[-1]
	out_f = out_path.split('/')[-1]

	in_azimuth = int(in_f.split('_')[0]) - 180
	out_azimuth = int(out_f.split('_')[0]) - 180

	return out_azimuth - in_azimuth
	
def generate_data_from_list(data_dict):
	while 1:
		#randomly sample a model
		models = data_dict.keys()
		rand_model = random.choice(models)

		#randomly sample a elevation from the chosen model
		elevations = data_dict[rand_model].keys()
		rand_elev = random.choice(elevations)

		in_img_path, out_img_path = random.sample(data_dict[rand_model][rand_elev], 2)
		
		view_transformation = get_azimuth_transformation(in_img_path, out_img_path)

		#TODO: figure out some way to bin these in 19 bins
		in_img = np.asarray(Image.open(in_img_path).convert('RGB'), dtype=np.uint8)
		out_img = np.asarray(Image.open(out_img_path).convert('RGB'), dtype=np.uint8)
		
		msk = imgMaskGen(out_img)

		yield ({'image_input': np.asarray([in_img]), 'view_input': view_transformation}, 
			{'sequential_3': np.asarray([out_img]), 'sequential_4': np.asarray([msk])})
		

def generate_data_dictionary(dataPath='../data/chairs/'):
	val_data_dict = {}
	train_data_dict = {}

	i=1
	for path,dirs,files in os.walk(dataPath):
		#print path
		for dr in dirs:
			print dr
			if dr!='model_views' and dr != '':
				drPath = path+'/'+dr
				if '//' not in drPath:
					#print drPath
					shutil.rmtree(drPath)
			#pruning complete
			elif dr =='model_views':
				train_data_dict[i]={}
				val_data_dict[i]={}

				inpath = os.path.join(dataPath,path[len(dataPath):]) + '/'+dr
				for files in os.walk(inpath):
					for fList in files:					
						for f in fList:
							if '.png' in f:
								#find elevation of file
								elevation = int(f.split('_')[1].replace('.png', ''))

								if elevation not in train_data_dict[i]:
									train_data_dict[i][elevation] = []
									val_data_dict[i][elevation] = []

								readLoc = inpath + '/'+f
								
								train_data_dict[i][elevation].append(readLoc)
								
				#assign 20% data to val_data_dict
				for e in train_data_dict[i]:
					d = train_data_dict[i][e]
					train_data_dict[i][e] = []

					random.shuffle(d)
					split_index = int(len(d)*0.8)
					train_data_dict[i][e].extend(d[0:split_index])
					val_data_dict[i][e].extend(d[split_index:])

				
				i += 1

	
	#pdb.set_trace()
	return train_data_dict, val_data_dict

