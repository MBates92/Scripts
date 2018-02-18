import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree

###############################################################################
'''Functions'''
###############################################################################

def raw_mom(x,n):
    return np.sum(x**n)/len(x)

###############################################################################

def central_mom(x,n):
    return np.sum((x-np.mean(x)**n))/len(x)

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

