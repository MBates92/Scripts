import numpy as np
import matplotlib.pyplot as plt

def TwoDalpha(alpha, pop):
    x = np.zeros(pop)
    y = np.zeros(pop)
    points = np.zeros((pop,2))
    for i in range(0,pop):
        R_r = np.random.uniform()
        R_phi = np.random.uniform()
        r = ((2.-alpha)*R_r/2.)**(1./(2.-alpha))
        phi = 2*np.pi*R_phi
        x[i] = r*np.cos(phi)
        y[i] = r*np.sin(phi)
    
    points[:,0] = x
    points[:,1] = y
    np.savetxt('NonFractalClusters/2D'+str(alpha)+'.txt',points)
    return points

def ThreeDalpha(alpha, pop):
    x = np.zeros(pop)
    y = np.zeros(pop)
    z = np.zeros(pop)
    points = np.zeros((pop,3))
    for i in range(0,pop):
        R_r = np.random.uniform()
        R_phi = np.random.uniform()
        R_theta = np.random.uniform()
        r = ((3.-alpha)*R_r/3.)**(1./(3.-alpha))
        theta = np.arccos(2.*R_theta - 1.)
        phi = 2*np.pi*R_phi
        x[i] = r*np.sin(theta)*np.cos(phi)
        y[i] = r*np.sin(theta)*np.sin(phi)
        z[i] = r*np.cos(theta)
    
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z
    
    np.savetxt('NonFractalClusters/3D'+str(alpha)+'.txt',points)
    return points

Zero = TwoDalpha(0.0, 500)
One = TwoDalpha(1.0, 500)

f,axarr = plt.subplots(1,2)
axarr[0].plot(Zero[:,0], Zero[:,1], 'bo')
axarr[0].set_title('2D0')
axarr[1].plot(One[:,0], One[:,1], 'bo')
axarr[1].set_title('2D1')

Zero = ThreeDalpha(0.0, 500)
One = ThreeDalpha(1.0, 500)
Two = ThreeDalpha(2.0, 500)
Three = ThreeDalpha(2.9, 500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Zero[:,0],Zero[:,1],Zero[:,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(One[:,0],One[:,1],One[:,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Two[:,0],Two[:,1],Two[:,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Three[:,0],Three[:,1],Three[:,2])

f,axarr = plt.subplots(1,4)
axarr[0].plot(Zero[:,0], Zero[:,1], 'bo')
axarr[0].set_title('3D0')
axarr[1].plot(One[:,0], One[:,1], 'bo')
axarr[1].set_title('3D1')
axarr[2].plot(Two[:,0], Two[:,1], 'bo')
axarr[2].set_title('3D2')
axarr[3].plot(Three[:,0], Three[:,1], 'bo')
axarr[3].set_title('3D2.9')