import numpy as np
import matplotlib.pyplot as plt
import time
import DeltaVar as dv
from Math import mean_square_error
from scipy.stats import gaussian_kde

L = 10**(np.linspace(-2,-0.5, 100))
L_range = 10**(np.linspace(-1.8,-1.4,100))
v = 1.5

H_actual = np.loadtxt("PeriodicNonExponentiatedFields/Testing/labels/labels.txt")

H_predicted = []
c=0
f,ax = plt.subplots(dpi=200)
for img in tqdm(os.listdir(DATADIR)):
    img_array = np.loadtxt(os.path.join(DATADIR,img))
    img_array /= 255.
    sigma_d = [dv.DeltaVar(img_array, l, v, periodicity = False) for l in L]
    sigma_d_range = [dv.DeltaVar(img_array, l, v, periodicity = False) for l in L_range]
    h = dv.HurstEstimator(sigma_d_range,L_range)
    H_predicted.append(h[0])
    if c<=10:
        ax.plot(np.log10(L),np.log10(sigma_d),label="{:.2f},{:.2f}".format(h[0],H_actual[c]))
    c += 1
ax.axvspan(np.log10(L_range[0]),np.log10(L_range[-1]), color = 'red', alpha = 0.2)
plt.legend()
plt.savefig("PeriodicNonExponentiatedFields/deltavar-{}.png".format(int(time.time())))

val_loss = mean_square_error(H_actual,H_predicted)

xy = np.vstack([H_actual,H_predicted.flatten()])
z = gaussian_kde(xy)(xy)

plt.figure(dpi = 200)
plt.scatter(H_actual,H_predicted,c=z)
plt.text(0.01, 0.95, 'MSE = '+ '{:.3g}'.format(val_loss))
plt.plot([0,1],[0,1],c='k')
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel(r'$H_{actual}$')
plt.ylabel(r'$H_{predicted}$')
plt.title('Unexponentiated-NonPeriodic-fbm-DV-{}'.format(int(time.time())))
plt.savefig("PeriodicNonExponentiatedFields/Periodic-fbm-DV-{}.png".format(int(time.time())))
