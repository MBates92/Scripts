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
    
    A = np.zeros((N,N), dtype = np.complex)
    if Seed != None:
        np.random.seed(int(Seed))
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
'''
X1 = SpectralSynthesis2D(N,0.0,Seed,1.0)
X2 = SpectralSynthesis2D(N,0.5,Seed,1.0)
X3 = SpectralSynthesis2D(N,1.0,Seed,1.0)

X4 = SpectralSynthesis2D(N,0.0,Seed,5.0)
X5 = SpectralSynthesis2D(N,0.5,Seed,5.0)
X6 = SpectralSynthesis2D(N,1.0,Seed,5.0)

X7 = SpectralSynthesis2D(N,0.0,Seed,10.0)
X8 = SpectralSynthesis2D(N,0.5,Seed,10.0)
X9 = SpectralSynthesis2D(N,1.0,Seed,10.0)
'''
###############################################################################
'''Plotting 2D Spectral Synthesis'''
###############################################################################
'''
f, axarr = plt.subplots(3,3,figsize=(1920/144, 1080/144), dpi=144)

axarr[0,0].imshow(X1)
axarr[0,0].set_title(r'$H=0.0,\sigma = 1.0$')
axarr[0,0].invert_yaxis()
axarr[1,0].imshow(X2)
axarr[1,0].set_title(r'$H=0.5,\sigma = 1.0$')
axarr[1,0].invert_yaxis()
axarr[2,0].imshow(X3)
axarr[2,0].set_title(r'$H=1.0,\sigma = 1.0$')
axarr[2,0].invert_yaxis()

axarr[0,1].imshow(X4)
axarr[0,1].set_title(r'$H=0.0,\sigma = 5.0$')
axarr[0,1].invert_yaxis()
axarr[1,1].imshow(X5)
axarr[1,1].set_title(r'$H=0.5,\sigma = 5.0$')
axarr[1,1].invert_yaxis()
axarr[2,1].imshow(X6)
axarr[2,1].set_title(r'$H=1.0,\sigma = 5.0$')
axarr[2,1].invert_yaxis()

axarr[0,2].imshow(X7)
axarr[0,2].set_title(r'$H=0.0,\sigma = 10.0$')
axarr[0,2].invert_yaxis()
axarr[1,2].imshow(X8)
axarr[1,2].set_title(r'$H=0.5,\sigma = 10.0$')
axarr[1,2].invert_yaxis()
axarr[2,2].imshow(X9)
axarr[2,2].set_title(r'$H=1.0,\sigma = 10.0$')
axarr[2,2].invert_yaxis()
plt.tight_layout()
plt.savefig('SpectralSynthesis/2DSignal.png')
'''
###############################################################################
'''Saving 2D Spectral Synthesis'''
###############################################################################
'''
np.savetxt('SpectralSynthesis/Signal/X1.txt', X1)
np.savetxt('SpectralSynthesis/Signal/X2.txt', X2)
np.savetxt('SpectralSynthesis/Signal/X3.txt', X3)

np.savetxt('SpectralSynthesis/Signal/X4.txt', X4)
np.savetxt('SpectralSynthesis/Signal/X5.txt', X5)
np.savetxt('SpectralSynthesis/Signal/X6.txt', X6)

np.savetxt('SpectralSynthesis/Signal/X7.txt', X7)
np.savetxt('SpectralSynthesis/Signal/X8.txt', X8)
np.savetxt('SpectralSynthesis/Signal/X9.txt', X9)
'''
###############################################################################
'''Implementing 3D Spectral Synthesis and Saving Signal'''
###############################################################################
'''
X1 = SpectralSynthesis3D(N,0.0,1.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H00_Sigma01', X1)
X2 = SpectralSynthesis3D(N,0.5,1.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H05_Sigma01', X2)
X3 = SpectralSynthesis3D(N,1.0,1.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H10_Sigma01', X3)

X4 = SpectralSynthesis3D(N,0.0,5.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H00_Sigma05', X4)
X5 = SpectralSynthesis3D(N,0.5,5.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H05_Sigma05', X5)
X6 = SpectralSynthesis3D(N,1.0,5.0)
np.save('SpectralSynthesis/3DSignal/X_H10_Sigma05', X6)

X7 = SpectralSynthesis3D(N,0.0,10.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H00_Sigma10', X7)
X8 = SpectralSynthesis3D(N,0.5,10.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H05_Sigma10', X8)
X9 = SpectralSynthesis3D(N,1.0,10.0,Seed)
np.save('SpectralSynthesis/3DSignal/X_H10_Sigma10', X9)
'''

X1 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma01.npy')
X2 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma01.npy')
X3 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma01.npy')

X4 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma05.npy')
X5 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma05.npy')
X6 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma05.npy')

X7 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma10.npy')
X8 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma10.npy')
X9 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma10.npy')

###############################################################################
'''Plotting 3D Signal'''
###############################################################################

shape1 = X1.shape
shape2 = X2.shape
shape3 = X3.shape

shape4 = X4.shape
shape5 = X5.shape
shape6 = X6.shape

shape7 = X7.shape
shape8 = X8.shape
shape9 = X9.shape

###############################################################################
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
for i in range(0,shape1[0]):
    for j in range(0,shape1[1]):
        for k in range(0,shape1[2]):
            ax.scatter(i,j,k, c='k', alpha = X1[i,j,k])
            print(str(i)+':'+str(j)+':'+str(k))

'''
for i in range(0,shape1[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X1[:,:,i], origin='lower')
    plt.title(r'$H=0.0,\sigma = 1.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma01/X_H00_Sigma01_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma01/X_H00_Sigma01_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma01/X_H00_Sigma01_'+
                    str(i)+'.png') 
        
for i in range(0,shape2[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X2[:,:,i], origin='lower')
    plt.title(r'$H=0.5,\sigma = 1.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma01/X_H05_Sigma01_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma01/X_H05_Sigma01_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma01/X_H05_Sigma01_'+
                    str(i)+'.png')
        
for i in range(0,shape3[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X3[:,:,i], origin='lower')
    plt.title(r'$H=1.0,\sigma = 1.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma01/X_H10_Sigma01_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma01/X_H10_Sigma01_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma01/X_H10_Sigma01_'+
                    str(i)+'.png')
        
###############################################################################
        
for i in range(0,shape4[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X4[:,:,i], origin='lower')
    plt.title(r'$H=0.0,\sigma = 5.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma05/X_H00_Sigma05_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma05/X_H00_Sigma05_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma05/X_H00_Sigma05_'+
                    str(i)+'.png')
        
for i in range(0,shape5[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X5[:,:,i], origin='lower')
    plt.title(r'$H=0.5,\sigma = 5.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma05/X_H05_Sigma05_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma05/X_H05_Sigma05_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma05/X_H05_Sigma05_'+
                    str(i)+'.png')
        
for i in range(0,shape6[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X6[:,:,i], origin='lower')
    plt.title(r'$H=1.0,\sigma = 5.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma05/X_H10_Sigma05_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma05/X_H10_Sigma05_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma05/X_H10_Sigma05_'+
                    str(i)+'.png')
        
###############################################################################       
              
for i in range(0,shape7[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X7[:,:,i], origin='lower')
    plt.title(r'$H=0.0,\sigma = 10.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma10/X_H00_Sigma10_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma10/X_H00_Sigma10_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H00_Sigma10/X_H00_Sigma10_'+
                    str(i)+'.png')
        
for i in range(0,shape8[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X8[:,:,i], origin='lower')
    plt.title(r'$H=0.5,\sigma = 10.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma10/X_H05_Sigma10_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma10/X_H05_Sigma10_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H05_Sigma10/X_H05_Sigma10_'+
                    str(i)+'.png')
        
for i in range(0,shape9[2]):
    plt.figure(figsize=(1920/144, 1080/144), dpi=144)
    plt.imshow(X9[:,:,i], origin='lower')
    plt.title(r'$H=1.0,\sigma = 10.0$')
    if i < 10:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma10/X_H10_Sigma10_00'+
                    str(i)+'.png')
    elif i>=10 and i<100:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma10/X_H10_Sigma10_0'+
                    str(i)+'.png')
    else:
        plt.savefig('SpectralSynthesis/3DSignal/H10_Sigma10/X_H10_Sigma10_'+
                    str(i)+'.png')
'''
###############################################################################