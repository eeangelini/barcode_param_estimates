import numpy as np
import numpy.random as rnd
import sys

import paramsInput as pin

import time # for profiling code

def dataGeneration(params,x0,times,R):
    K = x0.shape[0] # number of cell types = number of barcode lineages
    L = len(times)

    data = np.zeros((K,K,L,R))


    for r in range(R):
        data[:,:,0,r] = x0
        for l in range(L):
            if l != 0:
                data[:,:,l,r] = bdtCellsBarcode(params,data[:,:,l-1,r],times[l],times[l-1])
            else:
                data[:,:,l,r] = bdtCellsBarcode(params,x0,times[l],0)
                
    return data

def bdtCellsBarcode(params,x0,T,t0=0):
    K = x0.shape[0] # number of cell types = number of barcode lineages
    B = params[:,0] # vector of birth rates
    D = B - params[:,1] # vector of death rates
    nu = params[:,2:K+1] # matrix of transition rates (K x K-1)
    Q = np.zeros((K,K)) # augment: put zeros on diagonal, q_ij = nu_ij
    for i in range(K):
        Q[i,list(range(i))+list(range(i+1,K))] = nu[i] 

    x = x0.copy() # initialize cell type vector
    t = t0 # intitalize time

    sumAll = np.sum(B*np.sum(x,axis=1)) + np.sum(D*np.sum(x,axis=1))
    for i in range(K):
        sumAll += np.sum(np.sum(x[i,:])*Q[i,:])

    while x.sum() > 0 and t < T:
        x, t, sumAll = bdtStepBarcode(x,K,B,D,Q,t,sumAll)
        
    return x # return population vector

def bdtStepBarcode(x,K,B,D,Q,t,sumAll):
    # find time to next event - Exp(lambda = sum_all) 
    t += -(1/sumAll)*np.log(rnd.uniform()) # update time

    # find next event via cumulative sum over propensities
    r = rnd.uniform()
    mySum = 0
    breakVar = False
    for ii in range(3*K): # loop over cell type
        for jj in range(K): # loop over barcode lineages
            if ii < K: # birth event propensities
                mySum += B[ii]*x[ii,jj]
                if mySum > r*sumAll: # next event is type ii, lineage jj birth
                    x[ii,jj] += 1
                    sumAll += B[ii] + D[ii] + sum(Q[ii,:])
                    breakVar = True # for breaking outer loop
                    break

            elif K <= ii < 2*K: # death event propensities
                mySum += D[ii-K]*x[ii-K,jj]
                if mySum > r*sumAll: # next event is type ii-K1, lineage jj death
                    x[ii-K,jj] += -1
                    sumAll += -(B[ii-K] + D[ii-K] + sum(Q[ii-K,:]))
                    breakVar = True # for breaking outer loop
                    break

            else: # phenotype transition propensities
                for kk in range(K):
                    mySum += Q[ii-2*K,kk]*x[ii-2*K,jj]
                    if mySum > r*sumAll: # next event is phenotype transition ii-2K -> kk (within family jj)
                        x[ii-2*K,jj] += -1
                        x[kk,jj] += 1
                        sumAll += B[kk] + np.sum(Q[kk,:]) - B[ii-2*K] - np.sum(Q[ii-2*K,:])
                        breakVar = True # for breaking outer loop(s)
                        break

            if breakVar: # already found the next event, break loop over barcodes
                break

        if breakVar: # already found next event, break loop over cell types
            break

    return x,t,sumAll

def main(myStr):
    print('\n','Generating artificial data via birth-death process...')
    
    print('\n',' - Number of replicates: ', pin.R)
    print('\n',' - Time point(s): ', pin.T)
    print('\n',' - Initial condition:','\n', pin.X0,'\n')
    tic = time.perf_counter()
    data = dataGeneration(pin.paramsTrue,pin.X0,pin.T,pin.R)

    toc = time.perf_counter()

    np.save(myStr,data)

    print("Data saved to ",myStr,".npy",'\n')
    print(f"Run time: {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    myFileName = "myDataBarcodes"
    if len(sys.argv) > 1: # optional arguments passed in: file name
        myFileName = sys.argv[1]
    main("../data/"+myFileName)