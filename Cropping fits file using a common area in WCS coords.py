import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic

PATH = 'D:/Hi-GalTiles/FitsFiles/Reprojected_l222'

for file_no in range(0,7):

	fits_file = fits.open(os.path.join(PATH,os.listdir(PATH)[file_no]))

	hdu = fits_file[0]

	wcs = WCS(hdu.header)
	position = [222.1940905*u.deg,-0.9577370*u.deg]
	size = 1.8587050*u.deg
	left = position[0]+size/2
	right = position[0]-size/2

	position_pix = []
	position_pix = wcs.wcs_world2pix(position[0],position[1],0)
	left_pix = wcs.wcs_world2pix(left,position[1],0)
	right_pix = wcs.wcs_world2pix(right,position[1],0)

	position_pix = np.ceil(position_pix).astype(int)
	size_pix = np.ceil(right_pix[0]-left_pix[0]).astype(int)
	cutout = Cutout2D(hdu.data, position = position_pix, size = size_pix, wcs=wcs)

	hdu.data = cutout.data

	hdu.header.update(cutout.wcs.to_header())

	hdu.writeto('D:/Hi-GalTiles/FitsFiles/Cutout_l222/Cutout_{}'.format(os.listdir(PATH)[file_no]), overwrite=True)

