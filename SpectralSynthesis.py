import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
'''Functions'''
###############################################################################

def SpectralSynthesis2D(N,H,sigma,Seed=None):
    
    """
    Function that returns a 2D Fractal Surface given the Hurst Parameter, the 
    dimensions in 2D Euclidean Space, a random number seed and an amplification
    component
    
    ===Arguments===
    
    -N: integer
        size of 2D Euclidean space
    
    -H: integer
        Hurst Parameter
    
    -Seed: integer
        Seed for Random Number Generator
    
    -sigma: float
        Parameter that controls amplification of the signal
        
    ===Returns===
    
    -signal: float
        2D array of N size representing a Fractal Surface
    
    """
    
    if Seed != None:
        np.random.seed(int(Seed))  
        
    i, j = np.meshgrid(range(-int(N/2),int(N/2)+1),range(-int(N/2),int(N/2)+1))
    
    beta = 1+2*H

    k = (i*i+j*j)**(0.5)
    rad = np.where(k>0.0,k**(-(beta*0.5)),0.0)

    phase = 2*np.pi*np.random.random((N+1,N+1))

    phaseneg = phase[slice(None,None,-1),slice(None,None,-1)]
    phase = phase - phaseneg

    A = rad*np.cos(phase)+rad*np.sin(phase)*1j 
       
    A[0,:] = A[0,:] + A[-1,:]
    A = np.delete(A,-1,0)
    
    A[:,0] = A[:,0] + A[:,-1]
    A = np.delete(A,-1,1)  
    
    A = np.roll(A,int(N/2),axis = 0)
    A = np.roll(A,int(N/2),axis = 1)
    
    signal = np.fft.ifft2(A)
    signal = signal.real
    signal *= sigma/np.std(np.abs(signal))
    signal = np.exp(signal)
    signal = COM(signal)
    
    return signal


###############################################################################
    
def SpectralSynthesis3D(N,H,sigma,Seed=None):
    
    """
    Function that returns a 3D Fractal Surface given the Hurst Parameter, the 
    dimensions in 3D Euclidean Space, a random number seed and an amplification
    component
    
    ===Arguments===
    
    -N: integer
        size of 3D Euclidean space
    
    -H: integer
        Hurst Parameter
    
    -Seed: integer
        Seed for Random Number Generator
    
    -sigma: float
        Parameter that controls amplification of the signal
        
    ===Returns===
    
    -signal: float
        3D array of N size representing a Fractal Volume
    
    """
    
    if Seed != None:
        np.random.seed(int(Seed))  
        
    i, j, k = np.meshgrid(range(1,N+1),range(1,N+1),
                       range(1,N+1))
    shape = np.shape(i)
    print(shape)
    
    phase = 2*np.pi*np.random.random((N,N,N))
    rad = (i*i+j*j+k*k)**(-(2*H + 3)/4)*np.random.normal(size=(N,N,N))
    
    phaseneg = phase[[slice(None,None,-1)]*3]
    phase = phase - phaseneg
    
    A = rad*np.cos(phase)+rad*np.sin(phase)*1j
    
    X = np.fft.ifftn(A)
    signal = X.real
    signal = signal*sigma/np.std(np.abs(signal))
    signal = np.exp(signal)
    signal = COM(signal)
        
    return signal

###############################################################################

def COM(X):
    theta_i=[]
    for i in range(len(X.shape)):
        theta_i.append(np.arange(X.shape[i],
                                 dtype=np.double)*2.*np.pi/(X.shape[i]))
        
    # convert to grid format
    theta=np.meshgrid(*theta_i,indexing='ij')
    
    # loop over axes
    for i in range(len(theta)):
        
        # calculate shift
        xi=np.cos(theta[i])*np.abs(X)
        zeta=np.sin(theta[i])*np.abs(X)
        theta_bar=np.arctan2(-zeta.mean(),-xi.mean())+np.pi
        shift=np.int((X.shape[i])*0.5*theta_bar/np.pi)
        
        # shift array
        X=np.roll(X,int(X.shape[i]/2)-shift,i)
    
    return X

###############################################################################
'''Initialisation'''
###############################################################################
    
N=1000
Seed = 120

sample_H = np.random.random(10000)
sample_sigma = np.random.random(10000)*2 + 0.5

np.save('../SpectralSynthesis/2D/target/H_sample',sample_H)
np.save('../SpectralSynthesis/2D/target/sigma_sample',sample_sigma)

###############################################################################
'''Implementing'''
###############################################################################

for i in range(0,len(sample_H)):
    X = SpectralSynthesis2D(N,sample_H[i],sample_sigma[i])
    if i<10:
        np.save('../SpectralSynthesis/2D/Signal/X_0000'+str(i),X)
    elif i<100:
        np.save('../SpectralSynthesis/2D/Signal/X_000'+str(i),X)
    elif i<1000:
        np.save('../SpectralSynthesis/2D/Signal/X_00'+str(i),X)
    elif i<10000:
        np.save('../SpectralSynthesis/2D/Signal/X_0'+str(i),X)
    else:
        np.save('../SpectralSynthesis/2D/Signal/X_'+str(i),X)        
    print(i)