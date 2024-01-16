import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import paramsInput as pin

def bdODE(x0,T,A,numT=200):
    times = np.linspace(0,T,numT)
    odeX = np.zeros((x0.shape[0],numT))
    for j in range(numT):
        odeX[:,j] = expm((times[j]-times[0])*A).dot(x0)
    
    return times, odeX

bdX = np.load("../data/myDataBarcodes.npy")
bdXAvg = bdX.mean(axis = 3).sum(axis=1)
times1 = np.concatenate((np.zeros(1),pin.T))
bdXAvg = np.hstack((pin.X0.sum(axis=1)[:,None],bdXAvg))

times2, odeX = bdODE(pin.X0.sum(axis=1),pin.T[-1],pin.ATrue)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))

ax1.semilogy(times1,bdXAvg[0,:],'k--',linewidth=3,label='Gillespie mean')
ax1.semilogy(times2,odeX[0,:],'b-',linewidth=3,alpha = 0.5,label='ODE')
ax1.set_ylabel("Number of type 1 cells",fontsize=15)
ax1.set_xlabel("Time",fontsize=15)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend()

ax2.semilogy(times1,bdXAvg[1,:],'k--',linewidth=3,label='Gillespie mean')
ax2.semilogy(times2,odeX[1,:],'b-',linewidth=3,alpha = 0.5,label='ODE')
ax2.set_ylabel("Number of type 2 cells",fontsize=15)
ax2.set_xlabel("Time",fontsize=15)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.legend()

fig.suptitle('Validation of mean of Gillespie simulations (n = '+str(pin.R)+') against ODE model', fontsize=15)
fig.tight_layout(pad=5.0)
plt.subplots_adjust(top=0.89)
plt.savefig('../figs/bdValidate.png',dpi=500)
plt.show()

