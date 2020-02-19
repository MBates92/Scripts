import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import time
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

def box_plot(y_data, x_data, bins = 5):
	histogram = np.histogram(y_data,bins=bins)
	bins_true = histogram[1]
	ind = np.digitize(y_data,bins_true)
	bin_true_list = []
	bin_pred_list = []
	for i in range(1,bins+1):
		indices = np.where(ind==i)[0]
		bin_true_list.append(np.asarray(y_data)[indices])
		bin_pred_list.append(np.asarray(x_data)[indices])
	bin_true_list = np.asarray(bin_true_list)
	bin_pred_list = np.asarray(bin_pred_list)
	bin_diff_list = bin_pred_list - bin_true_list
	return [bin_diff_list,bins_true]

PATH = 'D:/ExponentiatedNonPeriodicNoisy/'

evaluations = np.loadtxt(PATH + 'evaluations.txt')
predictions = np.loadtxt(PATH + 'predictions.txt')

pickle_in = open(PATH + 'y.pickle','rb')
y_test = pickle.load(pickle_in)

total_loss = evaluations[0]
H_loss = evaluations[1]
sigma_loss = evaluations[2]

H_predicted = predictions[:,0]
sigma_predicted = predictions[:,1]

xy = np.vstack([y_test[0],H_predicted])
z = gaussian_kde(xy)(xy)

plt.figure(dpi = 250)
plt.scatter(y_test[0],H_predicted,c=z,s=5)
plt.colorbar()
plt.text(0.021, 0.95, 'RMSE = '+ '{:.3g}'.format(np.sqrt(H_loss)),bbox = dict(facecolor='white'))
plt.plot([0,1],[0,1],c='k')
plt.xlabel(r'$H_{true}$')
plt.ylabel(r'$H_{derived}$')
plt.tight_layout()
plt.grid()
plt.ylim([0,1])
plt.xlim([0,1])
plt.savefig(PATH + 'H-Performance.png')
plt.close()

plotting_data = box_plot(y_test[0],H_predicted)
bin_diff_list = plotting_data[0]
bins_true = plotting_data[1]

fig = plt.figure(1, dpi=250)
ax = fig.add_subplot(111)
bp=ax.boxplot(bin_diff_list)
ax.axhline()
ax.set_xticklabels(['{:.2f}'.format((bins_true[1]+bins_true[0])/2),'{:.2f}'.format((bins_true[2]+bins_true[1])/2),'{:.2f}'.format((bins_true[3]+bins_true[2])/2),'{:.2f}'.format((bins_true[4]+bins_true[3])/2),'{:.2f}'.format((bins_true[5]+bins_true[4])/2)])
ax.set_xlabel(r'$H_{true}$')
ax.set_ylabel(r'$H_{derived}-H_{true}$')
ax.set_ylim(-0.4,0.4)
fig.savefig(PATH + 'Box-Plot-H-Performance.png', bbox_inches='tight')
plt.close()

fig,(ax2,ax1,cax) = plt.subplots(nrows=3, figsize=(880/144, 2000/144), dpi=144,
								gridspec_kw={"height_ratios":[1,1, 0.05]})
im = ax1.scatter(y_test[0],H_predicted,c=z,s=9,cmap = 'inferno')
ax1.text(0.021, 0.95, 'RMSE = '+ '{:.3g}'.format(np.sqrt(H_loss)),bbox = dict(facecolor='white'))
ax1.plot([0,1],[0,1],c='k')
ax1.set_ylim([0,1])
ax1.set_xlim([0,1])
ax1.locator_params(axis='x',nbins=10)
ax1.locator_params(axis='y',nbins=10)
ax1.set_ylabel(r'$H_{derived}$')

bp=ax2.boxplot(bin_diff_list)
ax2.axhline()
ax2.set_xticklabels(['{:.2f}'.format((bins_true[1]+bins_true[0])/2),'{:.2f}'.format((bins_true[2]+bins_true[1])/2),'{:.2f}'.format((bins_true[3]+bins_true[2])/2),'{:.2f}'.format((bins_true[4]+bins_true[3])/2),'{:.2f}'.format((bins_true[5]+bins_true[4])/2)])
ax1.set_xlabel(r'$H_{true}$')
ax2.set_ylabel(r'$\overline{H_{derived}-H_{true}}$')
ax2.set_ylim(-0.4,0.4)

cbar = fig.colorbar(im,cax = cax,orientation='horizontal')
cbar.set_label(r'Kernel Density Estimation')
fig.savefig(PATH + 'Combi-H-Performance.png', bbox_inches='tight')
plt.close()

######

xy = np.vstack([y_test[1],sigma_predicted])
z = gaussian_kde(xy)(xy)

plt.figure(dpi = 250)
plt.scatter(y_test[1],sigma_predicted,c=z,s=5,cmap='inferno')
plt.colorbar()
plt.text(0.021, 0.95, 'RMSE = '+ '{:.3g}'.format(np.sqrt(sigma_loss)),bbox = dict(facecolor='white'))
plt.plot([0,3],[0,3],c='k')
plt.xlabel(r'$\sigma_{true}$')
plt.ylabel(r'$\sigma_{derived}$')
plt.tight_layout()
plt.grid()
plt.ylim([0,3])
plt.xlim([0,3])
plt.savefig(PATH + 'sigma-Performance.png')
plt.close()

plotting_data = box_plot(y_test[1],sigma_predicted)
bin_diff_list = plotting_data[0]
bins_true = plotting_data[1]

fig = plt.figure(1, dpi=250)
ax = fig.add_subplot(111)
bp=ax.boxplot(bin_diff_list)
ax.axhline()
ax.set_xticklabels(['{:.2f}'.format((bins_true[1]+bins_true[0])/2),'{:.2f}'.format((bins_true[2]+bins_true[1])/2),'{:.2f}'.format((bins_true[3]+bins_true[2])/2),'{:.2f}'.format((bins_true[4]+bins_true[3])/2),'{:.2f}'.format((bins_true[5]+bins_true[4])/2)])
ax.set_xlabel(r'$\sigma_{true}$')
ax.set_ylabel(r'$\sigma_{derived}-\sigma_{true}$')
ax.set_ylim(-0.4,0.4)
fig.savefig(PATH + 'Box-Plot-sigma-Performance.png', bbox_inches='tight')
plt.close()

fig,(ax2,ax1,cax) = plt.subplots(nrows=3, figsize=(880/144, 2000/144), dpi=144,
								gridspec_kw={"height_ratios":[1,1, 0.05]})
im = ax1.scatter(y_test[1],sigma_predicted,c=z,s=9,cmap = 'inferno')
ax1.text(0.021*3, 0.95*3, 'RMSE = '+ '{:.3g}'.format(np.sqrt(sigma_loss)),bbox = dict(facecolor='white'))
ax1.plot([0,3],[0,3],c='k')
ax1.set_ylim([0,3])
ax1.set_xlim([0,3])

ax1.set_ylabel(r'$\sigma_{derived}$')

bp=ax2.boxplot(bin_diff_list/3)
ax2.axhline()
ax2.set_xticklabels(['{:.2f}'.format((bins_true[1]+bins_true[0])/2),'{:.2f}'.format((bins_true[2]+bins_true[1])/2),'{:.2f}'.format((bins_true[3]+bins_true[2])/2),'{:.2f}'.format((bins_true[4]+bins_true[3])/2),'{:.2f}'.format((bins_true[5]+bins_true[4])/2)])
ax1.set_xlabel(r'$\sigma_{true}$')
ax2.set_ylabel(r'$\overline{\sigma_{derived}-\sigma_{true}}$')
ax2.set_ylim(-0.6,0.6)

cbar = fig.colorbar(im,cax = cax,orientation='horizontal')
cbar.set_label(r'Kernel Density Estimation')
fig.savefig(PATH + 'Combi-sigma-Performance.png', bbox_inches='tight')
plt.close()