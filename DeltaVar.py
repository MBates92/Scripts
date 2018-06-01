import numpy as np
from astropy.convolution import convolve

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

