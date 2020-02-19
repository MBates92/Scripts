import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as net
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm

###############################################################################
'''Functions'''
###############################################################################

def Pearson(y_actual,y_predict):
    Y_actual = np.mean(y_actual)
    Y_predict = np.mean(y_predict)
    diff_actual = y_actual-Y_actual
    diff_predict = y_predict-Y_predict
    numerator = np.sum(diff_actual*diff_predict)
    denom = np.sqrt(np.sum(diff_actual**2))*np.sqrt(np.sum(diff_predict**2))
    
    return numerator/denom

###############################################################################
    
def RMSE(y_actual,y_predict):
    
    return np.sqrt(np.sum((y_actual-y_predict)**2)/len(y_actual))

###############################################################################
'''
idx = np.array([12,13,14,15,16,17])

X = np.load('../SpectralSynthesis/2D/features.npy')
y = np.load('../SpectralSynthesis/2D/targets.npy').T

X_copy = np.copy(X)

#X_copy = X_copy[:,idx]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)

X_test_copy = np.copy(X_test)
X_train_copy = np.copy(X_train)

#X_test = X_test[:,idx]
#X_train = X_train[:,idx]

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = net.MLPRegressor(hidden_layer_sizes = (50,50,50), max_iter = 500)

mlp.fit(X_train,y_train)

print(mlp)

predictions = mlp.predict(X_test)

x_lin = np.linspace(np.amin(y_test[:,0]),np.amax(y_test[:,0]),100)
y_lin = x_lin

plt.figure()
plt.scatter(y_test[:,0],predictions[:,0])
plt.text(0.01, 0.95, 'r = '+ '{:.3g}'.format(Pearson(y_test[:,0],predictions[:,0])))
plt.text(0.01, 0.91, 'RMSE = '+ '{:.3g}'.format(RMSE(y_test[:,0],predictions[:,0])))
plt.plot(x_lin,y_lin,c='k')
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel(r'$H_{actual}$')
plt.ylabel(r'$H_{predicted}$')
plt.title(r'$H $, All Features')
plt.savefig('../SpectralSynthesis/MiscImages/hAll.png', dpi=1200)

x_lin = np.linspace(np.amin(y_test[:,1]),np.amax(y_test[:,1]),100)
y_lin = x_lin

plt.figure()
plt.scatter(y_test[:,1],predictions[:,1])
plt.text(0.51, 2.40, 'r = '+ '{:.3g}'.format(Pearson(y_test[:,1],predictions[:,1])))
plt.text(0.51, 2.32, 'RMSE = '+ '{:.3g}'.format(RMSE(y_test[:,1],predictions[:,1])))
plt.plot(x_lin,y_lin,c='k')
plt.ylim([0.5,2.5])
plt.xlim([0.5,2.5])
plt.xlabel(r'$\sigma_{actual}$')
plt.ylabel(r'$\sigma_{predicted}$')
plt.title(r'$\sigma $, All Features')
plt.savefig('../SpectralSynthesis/MiscImages/sigmaAll.png', dpi=1200)'''