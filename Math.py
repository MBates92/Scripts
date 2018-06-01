import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def lognormal(x, m, sigma):
    """
    Function that returns a lognormal function given a set of points, the mean 
    and the standard deviation
    
    ===Arguments===
    
    -x:  array of floats
    1D array containing a range of points
    
    -m: float
    mean of the lognormal
    
    -sigma: float
    standard deviation of the lognormal
    
    ===Returns===
    
    -array of floats
    points calculated for each x
    
    """
    numerator = (x-m)**2
    denominator = 2*sigma**2
    return np.exp(-numerator/denominator)

###############################################################################

def maxima2D(data, threshold = 0, size = 5):
    """
    Function that returns the positions of local maxima given a 2D function
    
    ===Arguments===
    
    -data:  float 
    image/function consisting of a 2D array of values
    
    -threshold: float
    number that determines the threshold value above which maxima are picked up
    
    -size: integer
    determines the size of the area over which the maxima are picked up
    
    ===Returns===
    
    -coords: array of floats
    coordinates of the maxima located by the algorithm
    
    """
    
    
    data_max = filters.maximum_filter(data, size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, size)
    diff = ((data_max - data_min) > threshold)
    
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    x,y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    
    coords = []
    coords.append(x)
    coords.append(y)
        
    coords = np.asarray(coords)
    
    return coords
    
###############################################################################