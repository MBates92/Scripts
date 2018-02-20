import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as net

X = np.load('../SpectralSynthesis/2D/features.npy')
y = np.load('../SpectralSynthesis/2D/targets.npy').T

n_samples = X.shape[0]
n_features = X.shape[1]
n_outputs = y.shape[1]

index = np.linspace(0,n_samples-1,n_samples, dtype = int)

np.random.shuffle(index)

index = index[:1000]

X_test = X[index,:]
y_test = y[index,:]

model = net.MLPRegressor()
model.fit(X,y)

y_predictions = model.predict(X_test)