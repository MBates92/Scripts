import numpy as np
import matplotlib.pyplot as plt
import SpectralSynthesis as ss
import DeltaVar as dv

def plot(N,sigma,H,n):
    X = []
    for i in range(n):
        X.append(ss.fBm(N,2,H,sigma))
        if i%10 == 0:
            print(i)
            
    X = np.asarray(X)
    
    L = 10**(np.linspace(np.log10(0.003),np.log10(0.1), 100))
    v = 1.5
    
    print(np.shape(X))
    
    deltavar = []
    for i in range(n):
        deltavar_i = []
        for j in range(len(L)):
            deltavar_i.append(dv.DeltaVar(X[i,:,:],L[j],v, method = 'convolve'))
        print(str(i))
        deltavar.append(deltavar_i)
    
    deltavar = np.asarray(deltavar)
    
    print(np.shape(deltavar))
    
    params = []
    for i in range((len(deltavar))):
        params.append(dv.HurstEstimator(deltavar[i],L))
        if i%10 == 0:
            print(i)
    params = np.asarray(params)
    
    H_est = params[:,0,:]
    std=H_est[:,1]
    H_est = H_est[:,0]
    var = std**2
    HMean = np.mean(H_est)
    varMean = np.mean(var)
    std = np.sqrt(varMean)
    
    plt.figure(figsize=(1080/144, 1080/144), dpi=144)
    ax = plt.axes()
    for i in range(len(deltavar)):
        plt.plot(np.log10(L),np.log10(deltavar[i])-np.log10(np.mean(deltavar[i])),'b', alpha = 0.7)
    plt.text(0.02,0.93,r'$H_{est}= $'+'%.3f'%HMean+r'$\pm %.3f$'%std, bbox=dict(facecolor='blue', alpha=0.3),transform = ax.transAxes)
    plt.title(r'Hundred Periodic Fields, N=1000, $H = '+str(H)+', \sigma = $' +str(sigma))
    plt.grid()
    plt.ylabel(r'$\log(\sigma^2_\Delta(L))$')
    plt.xlabel(r'$\log(L[pixels])$')
    if H == 0.0:
        Hsize = 'Zero'
    
    elif H == 0.5:
        Hsize = 'Half'
        
    elif H == 1.0:
        Hsize = 'One'    
    
    if sigma == 0.5:
        sigmasize = 'Small'
    
    elif sigma == 1.0:
        sigmasize = 'Mid'
        
    elif sigma == 2.0:
        sigmasize = 'Large'
    
    plt.savefig('../SpectralSynthesis/MiscImages/DeltaVarPeriodicH'+Hsize+'Sigma'+sigmasize+'Hundred',bbox_inches="tight")
    
H=np.array([0.0,0.5,1.0])
sigma = np.array([0.5,1.0,2.0])

for h in H:
    for s in sigma:
        plot(1000,s,h,100)