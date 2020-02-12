import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from Math import COM

def f(x, A, B):
    return A*x + B

def MexicanHat(r,L,v=1.5):
    core = 4/(np.pi*L**2) * np.exp(-r**2/(L/2)**2)
    annulus = 4 / (np.pi*L**2*(v**2-1))*(np.exp(-r**2/(v*L/2)**2)-
                   np.exp(-r**2/(L/2)**2))

    return core, annulus

def DeltaVar(X,L,v=1.5, periodicity = False):
    if periodicity == True:
        #constructing the filter grid
        shape = np.shape(X)
        N = shape[0]
        x = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        y = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        xv, yv = np.meshgrid(x,y)
        
        #calculate radial coordinate for each point
        r = np.sqrt(xv**2 + yv**2)
        r /=np.max(r)
        
        #produce filter
        core, annulus = MexicanHat(r,L,v)

        f = core - annulus
        
        f_fft = np.fft.fftn(f)
        X_fft = np.fft.fftn(X)
        
        prod = f_fft*X_fft
        
        g=np.fft.ifftn(prod).real

        deltavar = g.var()
    
    if periodicity == False:
        #padding the image
        shape = np.shape(X)
        img_size = shape[0]
        N = img_size*2
        x = np.zeros((N,N))
        x[:img_size,:img_size] = X
        X = x

        #constructing the filter grid
        x = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        y = np.linspace(-2**-0.5,2**-0.5,N, dtype=np.float64)
        xv, yv = np.meshgrid(x,y)
        
        #calculate radial coordinate for each point
        r = np.sqrt(xv**2 + yv**2)
        r /=np.max(r)

        r = np.roll(r,int(N/2),0)
        r = np.roll(r,int(N/2),1)
        
        #produce filter functions
        core, annulus = MexicanHat(r,L,v)
        core = core/np.sum(core)
        annulus = annulus/np.sum(annulus)

        #producing weightings
        zeros = np.zeros((N,N))
        ones = np.ones((img_size,img_size))
        zeros[:img_size,:img_size] = ones
        w = zeros

        #performing convolutions and trimming
        #FIELD (*) CORE
        prod = np.fft.fftn(X)*np.fft.fftn(core)
        G_core = np.fft.ifftn(prod).real
        G_core = G_core[:img_size,:img_size]

        #FIELD (*) ANNULUS
        prod = np.fft.fftn(X)*np.fft.fftn(annulus)
        G_annulus = np.fft.ifftn(prod).real
        G_annulus = G_annulus[:img_size,:img_size]

        #WEIGHTS (*) CORE
        prod = np.fft.fftn(w)*np.fft.fftn(core)
        W_core = np.fft.ifftn(prod).real
        W_core = W_core[:img_size,:img_size]

        #WEIGHTS (*) ANNULUS
        prod = np.fft.fftn(w)*np.fft.fftn(annulus)
        W_annulus = np.fft.ifftn(prod).real
        W_annulus = W_annulus[:img_size,:img_size]

        #Producing full convolution
        F = (G_core/W_core)-(G_annulus/W_annulus)

        #Calculating DeltaVariance
        F_mean = np.mean(F)
        W_tot = W_core*W_annulus
        deltavar = np.sum(((F-F_mean)**2)*(W_tot))/np.sum(W_tot)

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
    
    params = ([H,std_H, A,std_A,B,std_B])
    
    return params