import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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