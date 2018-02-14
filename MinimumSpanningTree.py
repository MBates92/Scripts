import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree

###############################################################################
'''Functions'''
###############################################################################

def MeanEdgeLength(points):
    
    xcoord,ycoord = MinSpanTree(points)
    
    N_total = points.shape[0]
    
    x1 = xcoord[:,0]
    x2 = xcoord[:,1]
    y1 = ycoord[:,0]
    y2 = ycoord[:,1]
    m = np.sqrt((x1-x2)**2+(y1-y2)**2)
    meanEdgeLength = np.mean(m)
    
    hull = ConvexHull(points)
    A = hull.area
    
    m_bar = meanEdgeLength*(N_total - 1)/np.sqrt((N_total*A))
    
    return m_bar
    
###############################################################################
    
###############################################################################

def MinSpanTree(points):
    
    """
    Function that takes a set of points and calculates the minimum spanning 
    tree,
    
    ===Arguments===
    
    -points: float
        2D array of points of size (N,2) containing x and y coordinates of
        N points
        
    ===Returns===
    
    -xcoord: float
        2D array of size (number,2) consisting of x-coordinates of edges
        
    -ycoord: float
        2D array of size (number,2) consisting of y-coordinates of edges
    
    """
    
    N = points.shape[0]
    tri = Delaunay(points)
    simplices = tri.simplices
    length_simplices = simplices.shape[0]
    graph_matrix = np.zeros((N,N))
    
    for i in range(0,length_simplices):
    	edge_1 = np.array([simplices[i,0],simplices[i,1]])
    	graph_matrix[simplices[i,0],simplices[i,1]] = np.sqrt((points[edge_1[0],
                  0]-points[edge_1[1],0])**2+ (points[edge_1[0],
                           1]-points[edge_1[1],1])**2)
    
    	edge_2 = np.array([simplices[i,0],simplices[i,2]])
    	graph_matrix[simplices[i,0],simplices[i,2]] = np.sqrt((points[edge_2[0],
                  0]-points[edge_2[1],0])**2+(points[edge_2[0],
                           1]-points[edge_2[1],1])**2)
    
    	edge_3 = np.array([simplices[i,1],simplices[i,2]])
    	graph_matrix[simplices[i,1],simplices[i,2]] = np.sqrt((points[edge_3[0],
                  0]-points[edge_3[1],0])**2+(points[edge_3[0],
                           1]-points[edge_3[1],1])**2)
    
    	print(length_simplices - i)
        
    print('Producing MST')
    Tcsr = minimum_spanning_tree(graph_matrix)
    Tcsr = Tcsr.tocoo()
    
    p1 = Tcsr.row
    p2 = Tcsr.col
    
    print('Linking data to MST')
    A = points[p1].T
    B = points[p2].T
    
    print('Producing coordinate system')
    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])
    
    return x_coords, y_coords
    
###############################################################################
'''Input data'''
###############################################################################

points1 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma01.txt')
points1 = np.delete(points1,2,1)
points2 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma01.txt')
points2 = np.delete(points2,2,1)
points3 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma01.txt')
points3 = np.delete(points3,2,1)

points4 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma05.txt')
points4 = np.delete(points4,2,1)
points5 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma05.txt')
points5 = np.delete(points5,2,1)
points6 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma05.txt')
points6 = np.delete(points6,2,1)

points7 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H00_Sigma10.txt')
points7 = np.delete(points7,2,1)
points8 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H05_Sigma10.txt')
points8 = np.delete(points8,2,1)
points9 = np.loadtxt('SpectralSynthesis/3DVariates/Variates_H10_Sigma10.txt')
points9 = np.delete(points9,2,1)

###############################################################################
'''Implementing MST'''
###############################################################################

x_coords1, y_coords1 = MinSpanTree(points1)
x_coords2, y_coords2 = MinSpanTree(points2)
x_coords3, y_coords3 = MinSpanTree(points3)

x_coords4, y_coords4 = MinSpanTree(points4)
x_coords5, y_coords5 = MinSpanTree(points5)
x_coords6, y_coords6 = MinSpanTree(points6)

x_coords7, y_coords7 = MinSpanTree(points7)
x_coords8, y_coords8 = MinSpanTree(points8)
x_coords9, y_coords9 = MinSpanTree(points9)

###############################################################################
'''Plotting MST'''
###############################################################################
'''
f, axarr = plt.subplots(3,3,figsize=(1920/144, 1080/144), dpi=144,
                        sharex=True, sharey=True)

axarr[0,0].plot(x_coords1, y_coords1, c='r', lw = 0.5)
axarr[0,0].scatter(points1[:,0],points1[:,1], c='k', s = 0.5)
axarr[0,0].set_title(r'$H=0.0,\sigma = 1.0$')
axarr[1,0].plot(x_coords2, y_coords2, c='r',lw = 0.5)
axarr[1,0].scatter(points2[:,0],points2[:,1], c='k', s = 0.5)
axarr[1,0].set_title(r'$H=0.5,\sigma = 1.0$')
axarr[2,0].plot(x_coords3, y_coords3, c='r',lw = 0.5)
axarr[2,0].scatter(points3[:,0],points3[:,1], c='k', s = 0.5)
axarr[2,0].set_title(r'$H=1.0,\sigma = 1.0$')

axarr[0,1].plot(x_coords4, y_coords4, c='r',lw = 0.5)
axarr[0,1].scatter(points4[:,0],points4[:,1], c='k', s = 0.5)
axarr[0,1].set_title(r'$H=0.0,\sigma = 5.0$')
axarr[1,1].plot(x_coords5, y_coords5, c='r',lw = 0.5)
axarr[1,1].scatter(points5[:,0],points5[:,1], c='k', s = 0.5)
axarr[1,1].set_title(r'$H=0.5,\sigma = 5.0$')
axarr[2,1].plot(x_coords6, y_coords6, c='r',lw = 0.5)
axarr[2,1].scatter(points6[:,0],points6[:,1], c='k', s = 0.5)
axarr[2,1].set_title(r'$H=1.0,\sigma = 5.0$')

axarr[0,2].plot(x_coords7, y_coords7, c='r',lw = 0.5)
axarr[0,2].scatter(points7[:,0],points7[:,1], c='k', s = 0.5)
axarr[0,2].set_title(r'$H=0.0,\sigma = 10.0$')
axarr[1,2].plot(x_coords8, y_coords8, c='r',lw = 0.5)
axarr[1,2].scatter(points8[:,0],points8[:,1], c='k', s = 0.5)
axarr[1,2].set_title(r'$H=0.5,\sigma = 10.0$')
axarr[2,2].plot(x_coords9, y_coords9, c='r',lw = 0.5)
axarr[2,2].scatter(points9[:,0],points9[:,1], c='k', s = 0.5)
axarr[2,2].set_title(r'$H=1.0,\sigma = 10.0$')
plt.tight_layout()
plt.savefig('SpectralSynthesis/MST.png')
'''