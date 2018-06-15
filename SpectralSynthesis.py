import numpy as np

###############################################################################
'''Functions'''
###############################################################################

def fBm(N, E, H, sigma=1., Seed=None, projected = None, axis = 0, exp=True):
    
    """
    Function that returns an E-D Fractal Surface given the edge length in
    Euclidean Space, the Euclidean dimension, and the Hurst Parameter.
    
    ===Arguments===
    
    -N: integer
        size of Euclidean space
    
    -E: integer
        Euclidean Dimension of the space
    
    -H: float
        Hurst Parameter
    
    -Seed: integer
        Seed for Random Number Generator
    
    -sigma: float
        Parameter that controls amplification of the signal
        
    ===Returns===
    
    -signal: float
        E-D array of N size representing a Fractal Surface
    
    """
    
    a = axis
    
    beta = E+2*H
    
    if Seed != None:
        np.random.seed(int(Seed))  
        
    range_list = []
    for i in range(0,E):
        range_list.append(range(-int(N/2),int(N/2)+1))
        
    c=0
    for i in np.meshgrid(*range_list):
        if c==0:
            k=i*i
        else:
            k += i*i
        c += 1
            
    k = k**0.5        
    rad = np.where(k>0.0,k**(-(beta*0.5)),0.0)
    rad /= ((rad**2).sum())**0.5
    phase = 2*np.pi*np.random.random(k.shape)

    phaseneg = phase[[slice(None,None,-1)]*E]
    phase -= phaseneg

    A = rad*np.cos(phase)+rad*np.sin(phase)*1j 
       
    for i in range(0,E):
        i_plus=[slice(None)]*E
        i_plus[i]=-1
        i_minus=[slice(None)]*E
        i_minus[i]=0
        A[i_minus] += A[i_plus]
        indices=[slice(None)]*E
        indices[i]=slice(None,-1)
        A = A[indices]
    
    offset = np.asarray(A.shape)/2
    offset = offset.astype(int)
    
    axis = np.arange(0,E)
    
    A = np.roll(A,offset,axis)
    
    X = np.fft.ifftn(A)
    X = X.real
    X *= sigma/np.std(X)
    
    if exp == True:
        X = np.exp(X)    
    
    if projected != None:
        if len(X.shape) != 3:
            if len(X.shape) == 2:
                print("Field is already in 2D")
                return COM(X)
            else:
                print("Please specify 3D field")
                print("Returning non-projected field")
                return COM(X)
        X = np.sum(X,axis=a)    
        X *= 1./np.std(X)  
        
    X = COM(X)
    
    return X


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
    