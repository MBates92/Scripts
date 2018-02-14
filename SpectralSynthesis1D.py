import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

N=500
Seed = np.random.random()*100

def SpectralSynthesis(N,H,Seed):
	A=np.zeros(N,dtype=np.complex_)
	beta=2*H+1
	np.random.seed(int(Seed))
	for i in range(0, int(N)):
		rad = (i+1)**(-beta/2)*np.random.normal()
		phase = 2*np.pi*np.random.random()
		A[i]=rad*np.cos(phase)+rad*np.sin(phase)*1j
	X = np.fft.ifft(A)
	return X

H1 = SpectralSynthesis(N,0.0,Seed)
H3 = SpectralSynthesis(N,0.3,Seed)
H5 = SpectralSynthesis(N,0.5,Seed)
H7 = SpectralSynthesis(N,0.7,Seed)
H9 = SpectralSynthesis(N,0.9,Seed)

f, axarr = plt.subplots(5, sharex=True)
axarr[0].plot(H1.real)
axarr[0].set_title(r'$H=0.0,\beta = 1.0$')
axarr[0].set_ylabel('x')
axarr[1].plot(H3.real)
axarr[1].set_title(r'$H=0.3,\beta = 1.6$')
axarr[1].set_ylabel('x')
axarr[2].plot(H5.real)
axarr[2].set_title(r'$H=0.5,\beta = 2.0$')
axarr[2].set_ylabel('x')
axarr[3].plot(H7.real)
axarr[3].set_title(r'$H=0.7,\beta = 2.4$')
axarr[3].set_ylabel('x')
axarr[4].plot(H9.real)
axarr[4].set_title(r'$H=1.0,\beta = 3.0$')
axarr[4].set_xlabel('t')
axarr[4].set_ylabel('x')
plt.tight_layout()
plt.show()