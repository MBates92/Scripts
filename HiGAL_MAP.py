import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from astropy.io import fits

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

image_file = 'D:/Bates_Lomax_Whitworth_Paper/Hi-Gal_DATA/l240_SPIRE_250um/l240_SPIRE_250um.fits'

hdu_list = fits.open(image_file)
hdu_list.info()

image_data = hdu_list[1].data

image_data = crop_center(image_data, 1051,1051)
image_data[np.where(np.isnan(image_data))] = np.min(np.nan_to_num(image_data))

trimmed_image = image_data - np.min(image_data)
trimmed_image = trimmed_image/np.max(trimmed_image)
image = np.log10(trimmed_image)

model = tf.keras.models.load_model("D:/Bates_Lomax_Whitworth_Paper/5-conv-512-channels-5-dense-100-epochs-1555584997.model")
IMG_SIZE = 128

H_array = np.zeros(np.shape(image))
sigma_array = np.zeros(np.shape(image))

for y in np.arange(np.shape(image)[1]):
    for x in np.arange(np.shape(image)[0]):
        sub_field = image[x-64:x+64,y-64:y+64]
        if np.shape(sub_field) == (128,128):
            m_1_field = np.mean(sub_field)
            s_1_field = np.std(sub_field)
            m_2_field = 0
            s_2_field = 1/4

            sub_field *= s_2_field/s_1_field
            sub_field += (m_2_field-m_1_field*s_2_field/s_1_field)
            
            X = np.array(sub_field).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            
            prediction = model.predict(X)
            H = prediction[0][0]
            sigma = prediction[1][0]
            
            H_array[x,y]=H
            sigma_array[x,y]=sigma
            
DATADIR = 'D:/Bates_Lomax_Whitworth_Paper/Hi-Gal_DATA/l240_SPIRE_250um/FBM_Map/'

np.savetxt(DATADIR+'H_array',H_array)
np.savetxt(DATADIR+'sigma_array',sigma_array)