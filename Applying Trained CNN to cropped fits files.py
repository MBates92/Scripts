import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from astropy.io import fits
import cv2
import csv

def beta(H, E=2):
	return E+2*H

def frac_D(H,E=2):
	return E+1-H

tile = 'l222'

model = tf.keras.models.load_model('D:/Bates_Lomax_Whitworth_Paper/BWL_CNN.model')

IMG_PATH = 'D:/Hi-GalTiles/FitsFiles/Cutout_{}'.format(tile)

text_file = [['Field Name','H','Beta','S','D']]

for file_no in range(0,7):

	fits_file = fits.open(os.path.join(IMG_PATH,os.listdir(IMG_PATH)[file_no]))

	hdu = fits_file[0]

	image_data = hdu.data

	image_data[np.where(np.isnan(image_data))] = np.min(np.nan_to_num(image_data))
	image_data = image_data - np.min(image_data)
	image_data[np.where(image_data == 0)] = np.sort(image_data.flatten()[np.where((image_data).flatten()!=0)])[0]/2

	IMG_SIZE = 128
	resized_img = cv2.resize(image_data, (IMG_SIZE,IMG_SIZE))

	m_1_field = np.mean(resized_img)
	s_1_field = np.std(resized_img)
	m_2_field = 0
	s_2_field = 1/4

	resized_img = resized_img*s_2_field/s_1_field
	resized_img = resized_img+(m_2_field-m_1_field*s_2_field/s_1_field)

	resized_img[np.where(resized_img>1.)] = 1.
	resized_img[np.where(resized_img<-1.)] = -1.

	plt.imshow(resized_img,origin='lower',cmap='hot')
	plt.colorbar()
	plt.title(os.listdir(IMG_PATH)[file_no])
	plt.savefig('D:/Hi-GalTiles/FitsFiles/Resized_{}/Resized_{}_map.png'.format(tile,os.listdir(IMG_PATH)[file_no]))
	plt.close()

	plt.figure()
	plt.hist(resized_img.flatten(),bins=100)
	plt.title(os.listdir(IMG_PATH)[file_no])
	plt.savefig('D:/Hi-GalTiles/FitsFiles/Resized_{}/Resized_{}_hist.png'.format(tile,os.listdir(IMG_PATH)[file_no]))
	plt.close()

	X = np.array(resized_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	prediction = model.predict(X)
	H = prediction[0][0]
	sigma = prediction[1][0]

	text_file.append([os.listdir(IMG_PATH)[file_no][19:-5], H[0], beta(H)[0], sigma[0], frac_D(H)[0]])

print(repr(text_file))

with open('D:/Hi-GalTiles/FitsFiles/{}_predictions.csv'.format(tile), 'w') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerows(text_file)

csvFile.close()