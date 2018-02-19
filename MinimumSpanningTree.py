import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations

###############################################################################
'''Functions'''
###############################################################################

def raw_mom(x,n):
    return np.sum(x**n)/len(x)

###############################################################################

def central_mom(x,n):
    return np.sum((x-np.mean(x))**n)*(1/len(x))

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

stars = np.load('../SpectralSynthesis/2D/Variates/X_62_99.npy')

x_coords, y_coords = MinSpanTree(stars)

MST_edges = np.sqrt((x_coords[0,:]-x_coords[1,:])**2+
                    (y_coords[0,:]-y_coords[1,:])**2)

Del_tri = tri.Triangulation(stars[:,0],stars[:,1])
Del_edges = Del_tri.edges
Del_edges = stars[Del_edges]

Del_x = Del_edges[:,:,0].T
Del_y = Del_edges[:,:,1].T

Del_edges = np.sqrt((Del_x[0,:]-Del_x[1,:])**2+
                    (Del_y[0,:]-Del_y[1,:])**2)

complete_edges = np.asarray(list((s[1],t[1]) for s,t in 
                        combinations(enumerate(stars),2)))

complete_x = complete_edges[:,0,:].T
complete_y = complete_edges[:,1,:].T

inj_x = np.copy(complete_x[1,:])
inj_y = np.copy(complete_y[0,:])

complete_x[1,:] = inj_y
complete_y[0,:] = inj_x

complete_edges = np.sqrt((complete_x[0,:]-complete_x[1,:])**2+
                         (complete_y[0,:]-complete_y[1,:])**2)

plt.figure()
plt.plot(complete_x,complete_y,c='r',lw=0.5)
plt.scatter(stars[:,0], stars[:,1], c= 'k', s = 2.0)
plt.show()

plt.figure()
plt.plot(Del_x,Del_y,c='r',lw=0.5)
plt.scatter(stars[:,0], stars[:,1], c= 'k', s = 2.0)
plt.show()

plt.figure()
plt.plot(x_coords,y_coords, c= 'r', lw = 0.5)
plt.scatter(stars[:,0], stars[:,1], c= 'k', s = 2.0)
plt.show()

MST_raw_moments = np.zeros(5)
MST_central_moments = np.zeros(5)

Del_raw_moments = np.zeros(5)
Del_central_moments = np.zeros(5)

for n in range(0,5):
    MST_raw_moments[n] = raw_mom(MST_edges,n)
    MST_central_moments[n] = central_mom(MST_edges,n)
    
    Del_raw_moments[n] = raw_mom(Del_edges,n)
    Del_central_moments[n] = central_mom(Del_edges,n)
    