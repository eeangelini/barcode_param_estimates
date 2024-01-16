import numpy as np
import scipy.optimize as opt
from scipy.linalg import logm

import paramsInput as pin

def approxRates(times,lineages,x0):
    L = len(times)
    K = lineages.shape[0]

    A = [] # estimate of A for each estimate of R(t): (1/t)*logm(R(t))

    for l in range(L):
        R = np.zeros((K,K)) 
        for i in range(K):
            for j in range(K):
                # R_ij(t) ~= lineage j cells in type i at t / initial number of lineage j cells
                R[i,j] = lineages[i,j,l]/x0[:,j].sum()
        A.append((1/times[l])*logm(R))

    return A

def mainBC():
    dataBC = np.load("../data/myDataBarcodes.npy")

    avgLineages = dataBC.mean(axis = 3) # take mean over replicates

    rateMatList = approxRates(pin.T,avgLineages,pin.X0) # get list of estimates of A (KxK matrices x L-1 time points)

    print(rateMatList)

    rateMatEst = sum(rateMatList)/pin.L # take time average of estimates of A

    rateMatTrue = pin.ATrue

    scale_est = 10**np.floor(np.log10(abs(rateMatEst))) # order of magnitude of estimates

    scale_true = 10**np.floor(np.log10(abs(rateMatTrue))) # true order of magnitude of parameters

    print("Number of replicates: ", pin.R,'\n')

    print("Number of time points: ", pin.L,'\n')
    
    print('Rate matrix estimate (time average):','\n',rateMatEst,'\n')

    print('True rate matrix:','\n',rateMatTrue,'\n')

    print("Estimated parameter scale: ",scale_est,'\n')

    print("True parameter scale: ",scale_true,'\n')

    # Report errors:

    print("Absolute error:",'\n',abs(rateMatEst-rateMatTrue),'\n')

    print("Maximum absolute error: ", abs(rateMatEst-rateMatTrue).max(),'\n')

    print("Relative error:",'\n',np.divide(abs(rateMatEst-rateMatTrue),abs(rateMatTrue)),'\n') 

    print("Maximum relative error:",'\n',np.divide(abs(rateMatEst-rateMatTrue),abs(rateMatTrue)).max(),'\n') 


if __name__ == '__main__':
    mainBC()
    

