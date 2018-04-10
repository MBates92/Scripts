import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

###############################################################################
'''Functions'''
###############################################################################

def IMF(m):
    return 0.076*np.exp(-((np.log10(m)-np.log10(0.25))**2)/(2*0.55**2))

###############################################################################
    
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

###############################################################################
    
def normalise(points):
    area = ConvexHull(points).volume
    points /= area
    return points

###############################################################################
    
def DTFE(points):

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

cluster_no = 7822
cluster_no = "%05d" % (cluster_no,)

stars = np.load('../SpectralSynthesis/2D/Variates/X_'+str(cluster_no)+'.npy')
stars = normalise(stars)

areas = DTFE(stars)

x=stars[:,0]
y=stars[:,1]

xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)

plt.figure()
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(stars[:,0],stars[:,1])