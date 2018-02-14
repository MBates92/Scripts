import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

###############################################################################
'''Functions'''
###############################################################################

def variates(signal, population, noise = True, noise_std = 0.5):
    
    """
    Function that takes a 2D signal, converts to PDF,
    samples random variates, adds Gaussian noise w. st.dev = 0.5
    
    ===Arguments===
    
    -signal: float
        input signal 
    
    -population: integer
        number of variates to be sampled
    
    -noise = True: Boolean
        whether to add Gaussian noise or not
    
    -noise_std = 0.5: float
        standard deviation of the Gaussian noise 
        
    ===Returns===
    
    -points: float
        2D array of size (number,2) consisting of coordinates of variates
    
    """
    if len(signal.shape) == 2:
        #convert signal to a normalised PDF
        PDF = signal/np.sum(signal)
        
        #flattens PDF into 1D array
        Flatten = PDF.flatten('C')
        
        #creates reference coordinates for each point on Flatten
        shape = signal.shape
        Y = np.repeat(np.linspace(0,shape[0]-1,shape[0]),shape[0])
        X = np.tile(np.linspace(0,shape[0]-1,shape[0]),shape[0])
        
        #xk is an index for each probability value in Flatten
        xk = np.linspace(0,len(X)-1,len(X))
        
        #creating rv_discrete object from xk and Flatten, and sampling
        custm = stats.rv_discrete(name = 'custm', values = (xk,Flatten))
        R=custm.rvs(size=population)
        
        #creating coordinate array
        points = np.zeros((len(R),2))
        points[:,0] = X[R]
        points[:,1] = Y[R]
        
        #implementing Gaussian noise condition
        if noise == True:
            for i in range(0,len(R)):
                points[i,0] += np.random.normal(0,noise_std)
                points[i,1] += np.random.normal(0,noise_std)
                
    if len(signal.shape) == 3:
        #convert signal to a normalised PDF
        PDF = signal/np.sum(signal)
        
        #flattens PDF into 1D array
        Flatten = PDF.flatten('C')
        
        #creates reference coordinates for each point on Flatten
        shape = signal.shape
        Z = np.repeat(np.linspace(0,shape[0]-1,shape[0]),shape[0]**2)
        X = np.tile(np.linspace(0,shape[0]-1,shape[0]),shape[0]**2)
        
        space2=np.zeros((shape[0],shape[0]))
        space1=np.linspace(0,shape[0]-1,shape[0])
        space2[:,:] = space1
        space2=space2.flatten('F')
        
        Y=np.zeros((shape[0]**2,shape[0]))
        for i in range(0,shape[0]):
            Y[:,i] = space2
        Y=Y.flatten('F')
        
        
        #xk is an index for each probability value in Flatten
        xk = np.linspace(0,len(X)-1,len(X))
        
        #creating rv_discrete object from xk and Flatten, and sampling
        custm = stats.rv_discrete(name = 'custm', values = (xk,Flatten))
        R=custm.rvs(size=population)
        
        #creating coordinate array
        points = np.zeros((len(R),3))
        points[:,0] = X[R]
        points[:,1] = Y[R]
        points[:,2] = Z[R]
        
        #implementing Gaussian noise condition
        if noise == True:
            for i in range(0,len(R)):
                points[i,0] += np.random.normal(0,noise_std)
                points[i,1] += np.random.normal(0,noise_std)
                points[i,2] += np.random.normal(0,noise_std)
    return points

###############################################################################
'''Initialisation of variables'''
###############################################################################

N=300 #number of stars

###############################################################################
'''Input 2D signal data'''
###############################################################################
'''
signal1 = np.loadtxt('SpectralSynthesis/2DSignal/X1.txt')
signal2 = np.loadtxt('SpectralSynthesis/2DSignal/X2.txt')
signal3 = np.loadtxt('SpectralSynthesis/2DSignal/X3.txt')

signal4 = np.loadtxt('SpectralSynthesis/2DSignal/X4.txt')
signal5 = np.loadtxt('SpectralSynthesis/2DSignal/X5.txt')
signal6 = np.loadtxt('SpectralSynthesis/2DSignal/X6.txt')

signal7 = np.loadtxt('SpectralSynthesis/2DSignal/X7.txt')
signal8 = np.loadtxt('SpectralSynthesis/2DSignal/X8.txt')
signal9 = np.loadtxt('SpectralSynthesis/2DSignal/X9.txt')
'''
###############################################################################
'''Input signal data'''
###############################################################################

signal1 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma01.npy')
signal2 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma01.npy')
signal3 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma01.npy')

signal4 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma05.npy')
signal5 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma05.npy')
signal6 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma05.npy')

signal7 = np.load('SpectralSynthesis/3DSignal/X_H00_Sigma10.npy')
signal8 = np.load('SpectralSynthesis/3DSignal/X_H05_Sigma10.npy')
signal9 = np.load('SpectralSynthesis/3DSignal/X_H10_Sigma10.npy')

###############################################################################
'''Variate sampling from signal data using variates function'''
###############################################################################

points1 = variates(signal1,N)
points2 = variates(signal2,N)
points3 = variates(signal3,N)

points4 = variates(signal4,N)
points5 = variates(signal5,N)
points6 = variates(signal6,N)

points7 = variates(signal7,N)
points8 = variates(signal8,N)
points9 = variates(signal9,N)

np.savetxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma01.txt',points1)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma01.txt',points2)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma01.txt',points3)

np.savetxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma05.txt',points4)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma05.txt',points5)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma05.txt',points6)

np.savetxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma10.txt',points7)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma10.txt',points8)
np.savetxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma10.txt',points9)

###############################################################################
'''Plotting 2D variates'''
###############################################################################
'''
f, axarr = plt.subplots(3,3,figsize=(1920/144, 1080/144), dpi=144,
                        sharex=True, sharey=True)

axarr[0,0].plot(points1[:,0],points1[:,1], 'b+')
axarr[0,0].set_title(r'$H=0.0,\sigma = 1.0$')
axarr[1,0].plot(points2[:,0],points2[:,1], 'b+')
axarr[1,0].set_title(r'$H=0.5,\sigma = 1.0$')
axarr[2,0].plot(points3[:,0],points3[:,1], 'b+')
axarr[2,0].set_title(r'$H=1.0,\sigma = 1.0$')

axarr[0,1].plot(points4[:,0],points4[:,1], 'b+')
axarr[0,1].set_title(r'$H=0.0,\sigma = 5.0$')
axarr[1,1].plot(points5[:,0],points5[:,1], 'b+')
axarr[1,1].set_title(r'$H=0.5,\sigma = 5.0$')
axarr[2,1].plot(points6[:,0],points6[:,1], 'b+')
axarr[2,1].set_title(r'$H=1.0,\sigma = 5.0$')

axarr[0,2].plot(points7[:,0],points7[:,1], 'b+')
axarr[0,2].set_title(r'$H=0.0,\sigma = 10.0$')
axarr[1,2].plot(points8[:,0],points8[:,1], 'b+')
axarr[1,2].set_title(r'$H=0.5,\sigma = 10.0$')
axarr[2,2].plot(points9[:,0],points9[:,1], 'b+')
axarr[2,2].set_title(r'$H=1.0,\sigma = 10.0$')
plt.tight_layout()
plt.savefig('SpectralSynthesis/2DVariates.png')
'''
###############################################################################
'''Plotting 3D variates'''
###############################################################################
'''
for i,j in [(points1,r'$H=0.0,\sigma = 1.0$'),
            (points2,r'$H=0.5,\sigma = 1.0$'),
            (points3,r'$H=1.0,\sigma = 1.0$'),
            (points4,r'$H=0.0,\sigma = 5.0$'),
            (points5,r'$H=0.5,\sigma = 5.0$'),
            (points6,r'$H=1.0,\sigma = 5.0$'),
            (points7,r'$H=0.0,\sigma = 10.0$'),
            (points8,r'$H=0.5,\sigma = 10.0$'),
            (points9,r'$H=1.0,\sigma = 10.0$')]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(i[:,0], i[:,1], i[:,2], c='b', marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(j)
    '''
###############################################################################
'''Plotting 2D projected 3D variates'''
###############################################################################

f, axarr = plt.subplots(3,3,figsize=(1920/144, 1080/144), dpi=144,
                        sharex=True, sharey=True)

axarr[0,0].plot(points1[:,0],points1[:,1], 'b+')
axarr[0,0].set_title(r'$H=0.0,\sigma = 1.0$')
axarr[1,0].plot(points2[:,0],points2[:,1], 'b+')
axarr[1,0].set_title(r'$H=0.5,\sigma = 1.0$')
axarr[2,0].plot(points3[:,0],points3[:,1], 'b+')
axarr[2,0].set_title(r'$H=1.0,\sigma = 1.0$')

axarr[0,1].plot(points4[:,0],points4[:,1], 'b+')
axarr[0,1].set_title(r'$H=0.0,\sigma = 5.0$')
axarr[1,1].plot(points5[:,0],points5[:,1], 'b+')
axarr[1,1].set_title(r'$H=0.5,\sigma = 5.0$')
axarr[2,1].plot(points6[:,0],points6[:,1], 'b+')
axarr[2,1].set_title(r'$H=1.0,\sigma = 5.0$')

axarr[0,2].plot(points7[:,0],points7[:,1], 'b+')
axarr[0,2].set_title(r'$H=0.0,\sigma = 10.0$')
axarr[1,2].plot(points8[:,0],points8[:,1], 'b+')
axarr[1,2].set_title(r'$H=0.5,\sigma = 10.0$')
axarr[2,2].plot(points9[:,0],points9[:,1], 'b+')
axarr[2,2].set_title(r'$H=1.0,\sigma = 10.0$')
plt.tight_layout()
plt.savefig('SpectralSynthesis/Projected3DVariates.png')
