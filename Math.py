import numpy as np
import matplotlib.pyplot as plt

def lognormal(x, m, sigma):
	numerator = (x-m)**2
	denominator = 2*sigma**2
	return np.exp(-numerator/denominator)