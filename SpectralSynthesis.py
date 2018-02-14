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
    
    i, j = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))
    
    A = np.zeros((N,N), dtype = np.complex)
    if Seed != None:
        np.random.seed(int(Seed))
    return i,j
        
'''        
    for i in range(0,int(N/2)+1):
    	for j in range(0,int(N/2)+1):

    		phase = 2*np.pi*np.random.random()
    
    		if (i!=0) or (j!=0):
    			rad = (i*i+j*j)**(-(H+1)/2)*np.random.normal()
    		else:
    			rad=0
    		A[i,j] = rad*np.cos(phase)+rad*np.sin(phase)*1j
    
    		if i==0:
    			i0 = 0
    		else:
    			i0=N-i
    		if j==0:
    			j0=0
    		else:
    			j0=N-j
    		A[i0,j0]=rad*np.cos(phase)-rad*np.sin(phase)*1j
     
    for i in range(1,int(N/2)):
    	for j in range(1,int(N/2)):
    		phase = 2*np.pi*np.random.random()
    		rad = (i*i+j*j)**(-(H+1)/2)*np.random.normal()
    		A[i,N-j] = rad*np.cos(phase)+rad*np.sin(phase)*1j
    		A[N-i,j] = rad*np.cos(phase)-rad*np.sin(phase)*1j
            
    np.savetxt('A.txt', A.imag)
    plt.figure()
    plt.imshow(np.log(np.abs(A.imag)))
    plt.colorbar()
    
    X = np.fft.ifft2(A)
    X_real = X.real
    signal = np.array(np.exp(X_real))
    signal = COM(signal)
    signal = signal-np.amin(signal)
    signal = signal/np.amax(signal)
    signal = signal**sigma
    
    return signal

'''

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
    
    A = np.zeros((N,N,N), dtype = np.complex)
    if Seed != None:
        np.random.seed(int(Seed))
    for i in range(0,int(N/2)+1):
        for j in range(0,int(N/2)+1):
            for k in range(0,int(N/2)+1):

                phase = 2*np.pi*np.random.random()

                if ((i!=0) or (j!=0)) or (k!=0):
                    rad = (i*i+j*j+k*k)**(-(2*H + 3)/4)*np.random.normal()
                else:
                    rad=0
                A[i,j,k] = rad*np.cos(phase)+rad*np.sin(phase)*1j
            
                if i==0:
                    i0 = 0
                else:
                    i0=N-i
                if j==0:
                    j0=0
                else:
                    j0=N-j
                if k==0:
                    k0=0
                else:
                    k0=N-k
                       
                A[i0,j0,k0]=rad*np.cos(phase)-rad*np.sin(phase)*1j
                
                print(str(H)+':'+str(sigma)+':'+str(i)+':'+str(j)+':'+str(k))
    
    for i in range(1,int(N/2)):
        for j in range(1,int(N/2)):
            for k in range(1,int(N/2)):
                phase = 2*np.pi*np.random.random()
                rad = (i*i+j*j+k*k)**(-(2*H + 3)/4)*np.random.normal()
                A[i,N-j,k] = rad*np.cos(phase)+rad*np.sin(phase)*1j
                A[N-i,j,N-k] = rad*np.cos(phase)-rad*np.sin(phase)*1j
                
                phase = 2*np.pi*np.random.random()
                rad = (i*i+j*j+k*k)**(-(2*H + 3)/4)*np.random.normal()
                A[N-i,N-j,k] = rad*np.cos(phase)+rad*np.sin(phase)*1j
                A[i,j,N-k] = rad*np.cos(phase)-rad*np.sin(phase)*1j
                
                phase = 2*np.pi*np.random.random()
                rad = (i*i+j*j+k*k)**(-(2*H + 3)/4)*np.random.normal()
                A[N-i,j,k] = rad*np.cos(phase)+rad*np.sin(phase)*1j
                A[i,N-j,N-k] = rad*np.cos(phase)-rad*np.sin(phase)*1j
                
                print(str(H)+':'+str(sigma)+':'+str(i)+':'+str(j)+':'+str(k))
            
    
    X = np.fft.ifftn(A)
    X_real = X.real
    signal = np.array(np.exp(X_real))
    signal = COM(signal)
    signal = signal-np.amin(signal)
    signal = signal/np.amax(signal)
    signal = signal**sigma
    
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
'''Initialisation of variables'''
###############################################################################
    
N=100
Seed = 120

###############################################################################
'''Implementing 2D Spectral Synthesis'''
###############################################################################

i,j = SpectralSynthesis2D(N,0.6,5.0,Seed)