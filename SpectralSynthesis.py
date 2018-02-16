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
    
    k = (i*i+j*j)**(0.5)
    rad = np.where(k>0,k**(-(H+1)),0.0)

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
    
    a = np.arange(36).reshape(6,6)
    shape = a.shape
    print(shape[0])
    print(a)
    a = np.roll(a,int(shape[0]/2),axis = 0)
    a = np.roll(a,int(shape[0]/2),axis = 1)
    print(a)

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

sample_H = np.linspace(0.0,1.0,100)
sample_sigma = np.linspace(0.0,2.0,100)

np.save('../SpectralSynthesis/2DSignal/H_sample',sample_H)
np.save('../SpectralSynthesis/2DSignal/sigma_sample',sample_sigma)

###############################################################################
'''Implementing'''
###############################################################################

X = SpectralSynthesis2D(N,1.0,0.1,Seed)

plt.figure()
plt.imshow(X)

'''
X = SpectralSynthesis2D(N,0.0,0.1,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.0,0.1')
plt.colorbar()

X = SpectralSynthesis2D(N,0.5,0.1,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.5,0.1')
plt.colorbar()

X = SpectralSynthesis2D(N,1.0,0.1,Seed)
plt.figure()
plt.imshow(X)
plt.title('1.0,0.1')
plt.colorbar()

X = SpectralSynthesis2D(N,0.0,1.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.0,1.0')
plt.colorbar()

X = SpectralSynthesis2D(N,0.5,1.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.5,1.0')
plt.colorbar()

X = SpectralSynthesis2D(N,1.0,1.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('1.0,1.0')
plt.colorbar()

X = SpectralSynthesis2D(N,0.0,2.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.0,2.0')
plt.colorbar()

X = SpectralSynthesis2D(N,0.5,2.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('0.5,2.0')
plt.colorbar()

X = SpectralSynthesis2D(N,1.0,2.0,Seed)
plt.figure()
plt.imshow(X)
plt.title('1.0,2.0')
plt.colorbar()
'''

'''
for i in range(0,len(sample_H)):
    for j in range(0,len(sample_sigma)):
        X = SpectralSynthesis2D(N,sample_H[i],sample_sigma[j])
        np.save('../SpectralSynthesis/2DSignal/X_'+str(i)+'_'+str(j),X)
        
        '''