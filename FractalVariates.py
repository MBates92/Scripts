import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

###############################################################################
'''Functions'''
###############################################################################

def variates(signal, population, noise = True, noise_std = 0.5, convert=False):
    
    """
    Function that takes a signal, converts to PDF,
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
            noise_vals = np.random.normal(0,noise_std,(len(R),2))
            points += noise_vals
                
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
                
        #project 3D points onto 2D plane        
        if convert == True:
            np.delete(points,2,1)
    return points

###############################################################################
'''Initialisation'''
###############################################################################

file_dir = '../SpectralSynthesis/2D/Signal/'
file_list = os.listdir(file_dir)

H_targets = np.load('../SpectralSynthesis/2D/target/H_sample.npy')
sigma_targets = np.load('../SpectralSynthesis/2D/target/sigma_sample.npy')

N = np.random.random_integers(100,300,len(H_targets))

###############################################################################
'''Implementation'''
###############################################################################

for i in range(len(file_list)):
    field = np.load(file_dir+file_list[i])
    stars = variates(field,N[i])
    name = file_list[i]
    name = name[:-4]
    np.save('../SpectralSynthesis/2D/Variates/'+file_list[i],stars)
    print(i)
