import numpy as np
from astropy.convolution import convolve
from scipy.optimize import curve_fit

def f(x, A, B):
    return A*x + B

def MexicanHat(r,L,v):
    core = 4/(np.pi*L**2) * np.exp(r**2/(L/2)**2)
    annulus = 4 / (np.pi*L**2*(v**2-1))*(np.exp(r**2/(v*L/2)**2)-
                   np.exp(r**2/(L/2)**2))
    return core - annulus

def DeltaVar(X,L,v):
    
    #constructing the filter grid
    shape = np.shape(X)
    N = shape[0]
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    xv, yv = np.meshgrid(x,y)
    
    #calculate radial coordinate for each point
    r = np.sqrt(xv**2 + yv**2)
    
    #produce filter
    f = MexicanHat(r,L,v)
    
    g = convolve(X,f,normalize_kernel = True)
    
    deltavar = 1/(2*np.pi) * np.mean(g**2)
    
    return deltavar

def DeltaVarFourier(X,L,v):
    
    #constructing the filter grid
    shape = np.shape(X)
    N = shape[0]
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xv, yv = np.meshgrid(x,y)
    
    #calculate radial coordinate for each point
    r = np.sqrt(xv**2 + yv**2)
    
    #produce filter
    f = MexicanHat(r,L,v)
    f /= np.sum(f)
    
    f_fft = np.fft.fft2(f)
    X_fft = np.fft.fft2(X)
    
    prod = X_fft.real*(np.abs(f_fft)**2)
    
    deltavar = np.sum(prod)/(2*np.pi)
    
    return deltavar

def LinearEstimator(deltavar,L,v):
    
    log_sigma_d = np.log10(deltavar)
    log_L = np.log10(L)
    
    smooth_width = 59
    x1 = np.linspace(-3,3,smooth_width)
    y1 = (4*x1**2 - 2) * np.exp(-x1**2) / smooth_width *8
    
    y_conv = np.convolve(log_sigma_d, y1, mode="same")
    
    linear_log_L = log_L[(y_conv >= -0.005) & (y_conv <= 0.005)]
    linear_log_sigma_d = log_sigma_d[(y_conv >= -0.005) & (y_conv <= 0.005)]
    
    A,B = curve_fit(f, linear_log_L, linear_log_sigma_d)[0]
    
    return A,B