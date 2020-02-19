import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from numpy.fft import rfft, irfft

def fbm(N,H,Seed=None):

	if Seed != None:
		np.random.seed(int(Seed))
		
	A=np.zeros(N,dtype=np.complex_)
	beta=2*H+1
	for i in range(0, int(N)):
		rad = (i+1)**(-beta/2)*np.random.normal()
		phase = 2*np.pi*np.random.random()
		A[i]=rad*np.cos(phase)+rad*np.sin(phase)*1j
	X = np.fft.ifft(A)
	return normalize(X)

def pink(N, state=None):

    X = np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1)
    S = np.sqrt(np.arange(len(X))+1.)
    y = (irfft(X/S)).real
    return normalize(y)

def white(N):
	return np.random.randn(N)

def ms(x):
    return (np.abs(x)**2.0).mean()

def normalize(y, x=None):
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt( x / ms(y) )