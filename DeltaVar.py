import numpy as np
from astropy.convolution import convolve_fft as convolve
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def f(x, A, B):
    return A*x + B

def MexicanHat(r,L,v):
    core = 4/(np.pi*L**2) * np.exp(-r**2/(L/2)**2)
    annulus = 4 / (np.pi*L**2*(v**2-1))*(np.exp(-r**2/(v*L/2)**2)-
                   np.exp(-r**2/(L/2)**2))
    return core - annulus

def DeltaVar(X,L,v, method = 'fourier'):
    
    if method == 'convolve':
        #constructing the filter grid
        shape = np.shape(X)
        N = shape[0]
        x = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        y = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        xv, yv = np.meshgrid(x,y)
        
        #calculate radial coordinate for each point
        r = np.sqrt(xv**2 + yv**2)
        r /= np.max(r)
        
        #produce filter
        f = MexicanHat(r,L,v)
        
        g = convolve(X,f, normalize_kernel = False, nan_treatment = 'fill', allow_huge=True)
        
        deltavar = np.mean(np.power(g,2), dtype=np.float64)

        deltavar /= (2*np.pi)
        
        return deltavar
    
    if method == 'fourier':
    
        #constructing the filter grid
        shape = np.shape(X)
        N = shape[0]
        x = np.linspace(0,1,N)
        y = np.linspace(0,1,N)
        xv, yv = np.meshgrid(x,y)
        
        xv = xv - np.max(xv)/2
        yv = yv - np.max(yv)/2
        
        #calculate radial coordinate for each point
        r = np.sqrt(xv**2 + yv**2)
        r /=np.max(r)
        
        #produce filter
        f = MexicanHat(r,L,v)
        
        f_fft = np.fft.fft2(f)
        X_fft = np.fft.fft2(X)
        
        prod = X_fft.real*(np.abs(f_fft)**2)
        
        deltavar = np.sum(prod)/(2*np.pi)
        
        return deltavar

def HurstEstimator(sigma_d,L,shift=0.5):
        
    log_sigma_d = np.log10(sigma_d)
    log_L = np.log10(L)
    
    y_spl = UnivariateSpline(log_L,log_sigma_d,s=0,k=4)
    y_spl_1d = y_spl.derivative(n=1)
    
    range_max1d = log_L[y_spl_1d(log_L)>(np.max(y_spl_1d(log_L))-shift)]
    
    linear_log_L = range_max1d
    linear_log_sigma_d = y_spl(range_max1d)
    
    cop,cov = curve_fit(f, linear_log_L, linear_log_sigma_d)
    
    A = cop[0]
    B = cop[1]
    std = np.sqrt(np.diag(cov))
    std_A = std[0]
    std_B = std[1]
    
    beta = A + 2
    H = (beta-2)/2
    
    std_H = std_A/2
    
    params = ([H,std_H],[A,std_A],[B,std_B])
    
    return params