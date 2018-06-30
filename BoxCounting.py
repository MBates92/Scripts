import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])

# =============================================================================
# def fractal_dimension(Z, threshold=0.9):
#     assert(len(Z.shape) == 2)
#     Z = (Z < threshold)
#     p = min(Z.shape)
#     n = 2**np.floor(np.log(p)/np.log(2))
#     n = int(np.log(n)/np.log(2))
#     sizes = 2**np.arange(n, 1, -1)
# 
#     counts = []
#     for size in sizes:
#         counts.append(boxcount(Z, size))
# 
#     coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
#     return -coeffs[0]
# =============================================================================

curve = plt.imread('out.png')

curve = rgb2gray(curve)

curve = curve<0.9
plt.imshow(curve)