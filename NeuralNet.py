import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as net
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.load('../SpectralSynthesis/2D/features.npy')
y = np.load('../SpectralSynthesis/2D/targets.npy').T

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = net.MLPRegressor(hidden_layer_sizes=(18,18,18),max_iter=500)

mlp.fit(X_train,y_train)

print(mlp)

predictions = mlp.predict(X_test)

x_lin = np.linspace(0,np.amax(y_test[:,0]),100)
y_lin = x_lin

plt.figure()
plt.scatter(y_test[:,0],predictions[:,0])
plt.plot(x_lin,y_lin)
plt.title('H predictions')
plt.show()

x_lin = np.linspace(0,np.amax(y_test[:,1]),100)
y_lin = x_lin

plt.figure()
plt.scatter(y_test[:,1],predictions[:,1])
plt.plot(x_lin,y_lin)
plt.title('Sigma predictions')
plt.show()