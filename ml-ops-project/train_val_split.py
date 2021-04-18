import os 
import shutil 

import numpy as np

def train_val_split(path_to_data_folder):

	os.makedirs(path_to_data_folder + '/val', exist_ok=True)
	TRAIN_FRAC = 0.7

	ARTIST_LIST = {i:name for i, name in enumerate(os.listdir(path_to_data_folder + 
															  '/train/'))}
	IMAGES_DIR = path_to_data_folder + '/train/'

	max_train_images = 0

	# создаем директорию с валидационной выборкой для каждого художника
	for artist in ARTIST_LIST.values():
	    os.makedirs(f'{path_to_data_folder}/val/{artist}/', exist_ok=True)

	    # считываем выборку картин художника
	    artist_path = f'{IMAGES_DIR}/{artist}/'
	    images_filename = os.listdir(artist_path)
	    
	    # выделяем часть картин для валидации
	    num_train = int(len(images_filename) * TRAIN_FRAC)
	    max_train_images = max(max_train_images, num_train)
	    val_images = images_filename[num_train:]

	    print(f'{artist} | train images = {num_train} | val images = {len(val_images)}')
	    
	    # сохраняем валидационную выборку
	    for image_filename in val_images:
	        source = f'{IMAGES_DIR}/{artist}/{image_filename}'
	        destination = f'{path_to_data_folder}/val/{artist}/{image_filename}'
	        shutil.copy(source, destination)
	        os.remove(source)


	 
if __name__ == '__main__':
	train_val_split('data')