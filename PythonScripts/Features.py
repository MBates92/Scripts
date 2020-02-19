import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

###############################################################################
'''Functions'''
###############################################################################

def hopkins(X):
    d = X.shape[1]
    n = len(X) 
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

###############################################################################
    
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

###############################################################################

def DTFE(points):
    area = ConvexHull(points).volume
    points /= np.sqrt(area)

    tri = Delaunay(points)

    simplices = tri.simplices
    
    simplices_area = []
    
    simplex_points = points[simplices]
    
    for i in range(0,len(simplices)):
        simplex_area = PolyArea(simplex_points[i,:,0],simplex_points[i,:,1])
        simplices_area.append(simplex_area)
    
    simplices_area = np.asarray(simplices_area)
    
    point_areas = []
    
    for i in range(0,len(points)):
        triangle_id = np.where(simplex_points == points[i,:])
        triangle_id = triangle_id[0]
        triangle_id = np.unique(triangle_id)
        triangle_areas = simplices_area[triangle_id]
        point_area = np.sum(triangle_areas)
        point_areas.append(point_area)
        
    point_areas = np.asarray(point_areas)
    
    return point_areas

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
    denominator = nth_root(central_mom(x,2),2)**n
    return nominator/denominator

###############################################################################

def Qmeasure(points):
    
    N_total = points.shape[0]
    
    MST_x, MST_y = MinSpanTree(points)
    
    m = np.sqrt((MST_x[0,:]-MST_x[1,:])**2+
                        (MST_y[0,:]-MST_y[1,:])**2)
    meanEdgeLength = np.mean(m)
    
    hull = ConvexHull(points)
    A = hull.volume
    
    m_bar = meanEdgeLength*(N_total - 1)/np.sqrt((N_total*A))
    
    complete_x, complete_y = completeGraph(points)
    s = np.sqrt((complete_x[0,:]-complete_x[1,:])**2+
                             (complete_y[0,:]-complete_y[1,:])**2)
    s_bar = np.mean(s)
    
    return m_bar/s_bar
    
###############################################################################
    
def completeGraph(points,plot = None, normalise=False):
    
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
    hull = ConvexHull(points)
    
    complete_edges = np.asarray(list((s[1],t[1]) for s,t in 
                            combinations(enumerate(points),2)))
    
    x_coords = complete_edges[:,0,:].T
    y_coords = complete_edges[:,1,:].T
    
    inj_x = np.copy(x_coords[1,:])
    inj_y = np.copy(y_coords[0,:])
    
    x_coords[1,:] = inj_y
    y_coords[0,:] = inj_x
    
    if normalise == True:
        x_coords/=np.sqrt(hull.volume)
        y_coords/=np.sqrt(hull.volume)     
    
    if plot == True:
        points/=np.sqrt(hull.volume)
        plt.figure()
        plt.plot(x_coords,y_coords, c='r',lw = 1,alpha=0.5)
        plt.scatter(points[:,0],points[:,1],c='k',s=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'Complete Graph')
        plt.grid(True)        
        plt.axis('equal')
        plt.savefig('../SpectralSynthesis/MiscImages/CompleteGraph.png', dpi=1200)
    
    return x_coords,y_coords
    
################################################v###############################
    
def DelaunayTriangulation(points,plot = None):
    
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
    hull = ConvexHull(points)
    
    Del_tri = tri.Triangulation(points[:,0],points[:,1])
    Del_edges = Del_tri.edges
    Del_edges = points[Del_edges]

    x_coords = Del_edges[:,:,0].T
    y_coords = Del_edges[:,:,1].T
    
    x_coords /= np.sqrt(hull.volume)
    y_coords /= np.sqrt(hull.volume)
    
    if plot == True:
        points/=np.sqrt(hull.volume)
        plt.figure()
        plt.plot(x_coords,y_coords, c='r',lw=1,alpha=0.5)
        plt.scatter(points[:,0],points[:,1],c='k',s = 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'Delaunay Triangulation')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('../SpectralSynthesis/MiscImages/DelaunayTri.png', dpi=1200)

    
    return x_coords,y_coords

###############################################################################

def MinSpanTree(points, plot = None,normalise=True):
    
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
    
    hull = ConvexHull(points)
    
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
    
    if normalise ==True:
        #Producing coordinate system
        x_coords = np.vstack([A[0], B[0]])/np.sqrt(hull.volume)
        y_coords = np.vstack([A[1], B[1]])/np.sqrt(hull.volume)
    else:
        #Producing coordinate system
        x_coords = np.vstack([A[0], B[0]])
        y_coords = np.vstack([A[1], B[1]])
        
    
    if plot == True:
        points/=np.sqrt(hull.volume)
        plt.figure()
        plt.plot(x_coords,y_coords, c='r',alpha=0.5)
        plt.scatter(points[:,0],points[:,1],c='k',s = 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'Minimum Spanning Tree')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('../SpectralSynthesis/MiscImages/MST.png', dpi=1200)
        
    
    return x_coords, y_coords

###############################################################################
    
def features(points, histogram = None, plots = None):
    
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
    
    feature_list = ['Euclidean Complete: Mean',
                    'Euclidean Complete: Variance',
                    'Euclidean Complete: Skewness',
                    'Euclidean Complete: Kurtosis',
                    'Euclidean Complete: Max Edge Length',
                    'Euclidean Complete: Min Edge Length',
                    'Euclidean Delaunay: Mean',
                    'Euclidean Delaunay: Variance',
                    'Euclidean Delaunay: Skewness',
                    'Euclidean Delaunay: Kurtosis',
                    'Euclidean Delaunay: Max Edge Length',
                    'Euclidean Delaunay: Min Edge Length',
                    'Euclidean MST: Mean',
                    'Euclidean MST: Variance',
                    'Euclidean MST: Skewness',
                    'Euclidean MST: Kurtosis',
                    'Euclidean MST: Max Edge Length',
                    'Euclidean MST: Min Edge Length',
                    'Manhattan Complete: Mean',
                    'Manhattan Complete: Variance',
                    'Manhattan Complete: Skewness',
                    'Manhattan Complete: Kurtosis',
                    'Manhattan Complete: Max Edge Length',
                    'Manhattan Complete: Min Edge Length',
                    'Manhattan Delaunay: Mean',
                    'Manhattan Delaunay: Variance',
                    'Manhattan Delaunay: Skewness',
                    'Manhattan Delaunay: Kurtosis',
                    'Manhattan Delaunay: Max Edge Length',
                    'Manhattan Delaunay: Min Edge Length',
                    'Manhattan MST: Mean',
                    'Manhattan MST: Variance',
                    'Manhattan MST: Skewness',
                    'Manhattan MST: Kurtosis',
                    'Manhattan MST: Max Edge Length',
                    'Manhattan MST: Min Edge Length',
                    'Hopkins Statistic',
                    'Areas: Mean',
                    'Areas: Variance',
                    'Areas: Skewness',
                    'Areas: Kurtosis',
                    'Areas: Max Area',
                    'Areas: Min Area',
                    'Number of Stars',
                    'Q measure']
    
    features = np.zeros(len(feature_list))
    
    MST_x, MST_y = MinSpanTree(points,plots)
    Euclidean_MST_edges = np.sqrt((MST_x[0,:]-MST_x[1,:])**2+
                        (MST_y[0,:]-MST_y[1,:])**2)

    Manhattan_MST_edges = np.abs(MST_x[0,:]-MST_x[1,:])+np.abs(MST_y[0,:]-MST_y[1,:])
    
    Del_x, Del_y = DelaunayTriangulation(points,plots)
    Euclidean_Del_edges = np.sqrt((Del_x[0,:]-Del_x[1,:])**2+
                        (Del_y[0,:]-Del_y[1,:])**2)
    Manhattan_Del_edges = np.abs(Del_x[0,:]-Del_x[1,:])+np.abs(Del_y[0,:]-Del_y[1,:])    
    
    complete_x, complete_y = completeGraph(points,plots)
    Euclidean_complete_edges = np.sqrt((complete_x[0,:]-complete_x[1,:])**2+
                             (complete_y[0,:]-complete_y[1,:])**2)
    Manhattan_complete_edges = np.abs(complete_x[0,:]-complete_x[1,:])+np.abs(complete_y[0,:]-complete_y[1,:])

    Area = DTFE(points)

    if histogram == 1:
        plt.figure()
        plt.hist(Euclidean_MST_edges,50, normed=True, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Euclidean MST')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/EuclideanMST.png', dpi=1200)
        
        plt.figure()
        plt.hist(Manhattan_MST_edges, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Manhattan MST')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/ManhattanMST.png', dpi=1200)
    
        
        plt.figure()
        plt.hist(Euclidean_Del_edges, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Euclidean Delaunay')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/EuclideanDel.png', dpi=1200)
        
        plt.figure()
        plt.hist(Manhattan_Del_edges, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Manhattan Delaunay')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/ManhattanDel.png', dpi=1200)
        
    
        plt.figure()
        plt.hist(Euclidean_complete_edges, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Euclidean Complete')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/EuclideanComplete.png', dpi=1200)
        
        plt.figure()
        plt.hist(Manhattan_complete_edges, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Edge Lengths')
        plt.title(r'Manhattan Complete')
        plt.grid(True)
        plt.savefig('../SpectralSynthesis/MiscImages/ManhattanComplete.png', dpi=1200)    

    features[0] = raw_mom(Euclidean_complete_edges,1)
    features[1] = central_mom(Euclidean_complete_edges,2)
    features[2] = normalised_mom(Euclidean_complete_edges,3)
    features[3] = normalised_mom(Euclidean_complete_edges,4)
    features[4] = np.amax(Euclidean_complete_edges)
    features[5] = np.amin(Euclidean_complete_edges)
    
    features[6] = raw_mom(Euclidean_Del_edges,1)
    features[7] = central_mom(Euclidean_Del_edges,2)
    features[8] = normalised_mom(Euclidean_Del_edges,3)
    features[9] = normalised_mom(Euclidean_Del_edges,4)
    features[10] = np.amax(Euclidean_Del_edges)
    features[11] = np.amin(Euclidean_Del_edges)
    
    features[12] = raw_mom(Euclidean_MST_edges,1)
    features[13] = central_mom(Euclidean_MST_edges,2)
    features[14] = normalised_mom(Euclidean_MST_edges,3)
    features[15] = normalised_mom(Euclidean_MST_edges,4)
    features[16] = np.amax(Euclidean_MST_edges)
    features[17] = np.amin(Euclidean_MST_edges)    
    
    features[18] = raw_mom(Manhattan_complete_edges,1)
    features[19] = central_mom(Manhattan_complete_edges,2)
    features[20] = normalised_mom(Manhattan_complete_edges,3)
    features[21] = normalised_mom(Manhattan_complete_edges,4)
    features[22] = np.amax(Manhattan_complete_edges)
    features[23] = np.amin(Manhattan_complete_edges)
    
    features[24] = raw_mom(Manhattan_Del_edges,1)
    features[25] = central_mom(Manhattan_Del_edges,2)
    features[26] = normalised_mom(Manhattan_Del_edges,3)
    features[27] = normalised_mom(Manhattan_Del_edges,4)
    features[28] = np.amax(Manhattan_Del_edges)
    features[29] = np.amin(Manhattan_Del_edges)
    
    features[30] = raw_mom(Manhattan_MST_edges,1)
    features[31] = central_mom(Manhattan_MST_edges,2)
    features[32] = normalised_mom(Manhattan_MST_edges,3)
    features[33] = normalised_mom(Manhattan_MST_edges,4)
    features[34] = np.amax(Manhattan_MST_edges)
    features[35] = np.amin(Manhattan_MST_edges)
    
    features[36] = hopkins(points)
    
    features[37] = raw_mom(Area,1)
    features[38] = central_mom(Area,2)
    features[39] = normalised_mom(Area,3)
    features[40] = normalised_mom(Area,4)
    features[41] = np.amax(Area)
    features[42] = np.amin(Area)
    
    features[43] = len(points)
    features[44] = Qmeasure(points)
    
    return features