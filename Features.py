import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations

###############################################################################
'''Functions'''
###############################################################################

def nth_root(x,n):
    y = np.absolute(x)**(1/n)
    y *= np.sign(x)**n
    return y

###############################################################################

def raw_mom(x,n):
    return np.sum(x**n)/len(x)

###############################################################################

def central_mom(x,n):
    return np.sum((x-np.mean(x))**n)*(1/len(x))

###############################################################################
    
def normalised_mom(x,n):
    nominator = central_mom(x,n)
    denominator = nth_root(central_mom(x,n),2)**n
    return nominator/denominator

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
    
def completeGraph(points):
    
    """
    Function that take a set of points and returns the edge coordinates of 
    their Complete Graph.
    
    ===Arguments===
    
    -points: float
        2D array of points of size (N,2) containing the x and y coordinates of
        N points
        
    ===Returns===
    
    -xcoord: float
        2D array of size (M,2) consisting of x-coordinates of M number of edges
        where M = (1/2)*N*(N-1)
        
    -ycoord: float
        2D array of size (M,2) consisting of y-coordinates of M number of edges
        where M = (1/2)*N*(N-1)
    
    """
    
    complete_edges = np.asarray(list((s[1],t[1]) for s,t in 
                            combinations(enumerate(points),2)))
    
    x_coords = complete_edges[:,0,:].T
    y_coords = complete_edges[:,1,:].T
    
    inj_x = np.copy(x_coords[1,:])
    inj_y = np.copy(y_coords[0,:])
    
    x_coords[1,:] = inj_y
    y_coords[0,:] = inj_x
    
    return x_coords,y_coords
    
###############################################################################
    
def DelaunayTriangulation(points):
    
    """
    Function that take a set of points and returns the edge coordinates of 
    their Delaunay Triangulation.
    
    ===Arguments===
    
    -points: float
        2D array of points of size (N,2) containing the x and y coordinates of
        N points
        
    ===Returns===
    
    -xcoord: float
        2D array of size (M,2) consisting of x-coordinates of M number of edges
        
    -ycoord: float
        2D array of size (M,2) consisting of y-coordinates of M number of edges
    
    """
    
    Del_tri = tri.Triangulation(points[:,0],points[:,1])
    Del_edges = Del_tri.edges
    Del_edges = stars[Del_edges]

    x_coords = Del_edges[:,:,0].T
    y_coords = Del_edges[:,:,1].T
    
    return x_coords,y_coords

###############################################################################

def MinSpanTree(points):
    
    """
    Function that takes a set of points and returns the edge coordinates of 
    their Minimum Spanning Tree.
    
    ===Arguments===
    
    -points: float
        2D array of points of size (N,2) containing x and y coordinates of
        N points
        
    ===Returns===
    
    -xcoord: float
        2D array of size (N-1,2) consisting of x-coordinates of N-1 number of
        edges
        
    -ycoord: float
        2D array of size (N-1,2) consisting of y-coordinates of N-1 number of 
        edges
    
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
    
        
    #'Producing MST
    Tcsr = minimum_spanning_tree(graph_matrix)
    Tcsr = Tcsr.tocoo()
    
    p1 = Tcsr.row
    p2 = Tcsr.col
    
    #Linking data to MST
    A = points[p1].T
    B = points[p2].T
    
    #Producing coordinate system
    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])
    
    return x_coords, y_coords

###############################################################################
    
def features(points):
    
    """
    Function that takes a set of points and calculates features of the complete
    graph, the Delaunay Triangulation, and Minimum Spanning Tree.
    
    ===Arguments===
    
    -points: float
        2D array of points of size (N,2) containing x and y coordinates of
        N points
        
    ===Returns===
    
    -features: float
        1D array containing a number of features calculated from the complete 
        graph, Delaunay Triangulation, and the Minimum Spanning Tree
    
    """
    
    feature_list = ['Complete: Mean',
                    'Complete: Variance',
                    'Complete: Skewness',
                    'Complete: Kurtosis',
                    'Complete: Max Edge Length',
                    'Complete: Min Edge Length',
                    'Delaunay: Mean',
                    'Delaunay: Variance',
                    'Delaunay: Skewness',
                    'Delaunay: Kurtosis',
                    'Delaunay: Max Edge Length',
                    'Delaunay: Min Edge Length',
                    'MST: Mean',
                    'MST: Variance',
                    'MST: Skewness',
                    'MST: Kurtosis',
                    'MST: Max Edge Length',
                    'MST: Min Edge Length']
    
    features = np.zeros(len(feature_list))
    
    MST_x, MST_y = MinSpanTree(points)
    MST_edges = np.sqrt((MST_x[0,:]-MST_x[1,:])**2+
                        (MST_y[0,:]-MST_y[1,:])**2)
    
    Del_x, Del_y = DelaunayTriangulation(points)
    Del_edges = np.sqrt((Del_x[0,:]-Del_x[1,:])**2+
                        (Del_y[0,:]-Del_y[1,:])**2)
    
    complete_x, complete_y = completeGraph(points)
    complete_edges = np.sqrt((complete_x[0,:]-complete_x[1,:])**2+
                             (complete_y[0,:]-complete_y[1,:])**2)
    
    complete_edges /= np.amax(complete_edges)
    Del_edges /= np.amax(complete_edges)
    MST_edges /= np.amax(complete_edges)
    
    features[0] = raw_mom(complete_edges,1)
    features[1] = central_mom(complete_edges,2)
    features[2] = normalised_mom(complete_edges,3)
    features[3] = normalised_mom(complete_edges,4)
    features[4] = np.max(complete_edges)
    features[5] = np.min(complete_edges)
    
    features[6] = raw_mom(Del_edges,1)
    features[7] = central_mom(Del_edges,2)
    features[8] = normalised_mom(Del_edges,3)
    features[9] = normalised_mom(Del_edges,4)
    features[10] = np.max(Del_edges)
    features[11] = np.min(Del_edges)
    
    features[12] = raw_mom(MST_edges,1)
    features[13] = central_mom(MST_edges,2)
    features[14] = normalised_mom(MST_edges,3)
    features[15] = normalised_mom(MST_edges,4)
    features[16] = np.max(MST_edges)
    features[17] = np.min(MST_edges)
    
    return features

###############################################################################
'''Input and Initialisation'''
###############################################################################

file_dir = '../SpectralSynthesis/2D/Variates/'
file_list = os.listdir(file_dir)

H_targets = np.load('../SpectralSynthesis/2D/target/H_sample.npy')
sigma_targets = np.load('../SpectralSynthesis/2D/target/sigma_sample.npy')

X = []
y=[]

###############################################################################
'''Implementation'''
###############################################################################

for i in range(len(file_list)):
    stars = np.load(file_dir+file_list[i])
    X_i = features(stars)
    X.append(X_i)
    print(i)

X = np.asarray(X)
y.append(H_targets)
y.append(sigma_targets)
y=np.asarray(y)
np.save('../SpectralSynthesis/2D/features',X)
np.save('../SpectralSynthesis/2D/targets',y)