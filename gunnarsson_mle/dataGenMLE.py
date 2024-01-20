import numpy as np
import numpy.random as rnd

import paramsInputMLE as pin
import time, sys
from joblib import Parallel, delayed # for parallel loops

def dataGeneration(params,X0,T,R):
    K = params.shape[0] # number of cell types
    I = X0.shape[0] # number of initial conditions OR number of barcode lineages
    L = len(T) # number of endpoints for collecting data

    data = np.zeros((I,K,L,R))


    for i in range(I):
        for r in range(R):
            tempFunc = lambda idx: bdtCells(params,X0[i,:],T[idx])
            data[i,:,:,r] = np.array(Parallel(n_jobs=-1)(delayed(tempFunc)(l) for l in range(L))).T
                        
    return data

def bdtCells(params,x0,T):
    '''documentation'''

    K = x0.shape[0]
    x = x0.copy() # initialize cell type vector
    t = 0 # initialize time

    params[:,1] = params[:,0]-params[:,1] # need death rate d instead of net growth rate lambda = b-d --> d = b-lambda

    a = np.sum(params, axis=1) # propensities of next event happening to each type of cell
    normedParams = np.divide(params,a[:,None]) # divide parameters by these propensities
    sumNormed = np.zeros((K,K+1))
    for j in range(K+1): # cumulative sum for event type
        sumNormed[:,j] = np.sum(normedParams[:,0:j+1],axis=1)


    while x.sum() > 0 and t < T:
        totalRates = x*a

        cumSum = 0
        t += -1/(np.sum(totalRates))*np.log(rnd.uniform())
        
        u = rnd.uniform()
        for i in range(K): # first step of next event: which type cell
            cumSum = cumSum + totalRates[i]
            if u < cumSum/np.sum(totalRates): # type i cell "does" next event
                numType = i
                break

        u = rnd.uniform()
        for j in range(K+1): # second step of next event: which event (birth/death/transition)
            if u < sumNormed[numType,j]: # type j event (j = 0: birth, j = 1: death, j >= 2: transition)
                numEvent = j
                break
        
        if numEvent == 0: # numType birth
            x[numType] += 1
        elif numEvent == 1: # numType death
            x[numType] += -1
        elif numEvent-2 < numType: #  Has to do with paramter matrix structure: if index (numType,numEvent-2) is below main diagonal of the nu_ij's (submatrix column 2 ->end), transition event is numType -> numEvent-2
            x[numType] += -1
            x[numEvent-2] += 1
        else: # else, index (numType,numEvent-2) is on or above main diagonal of the nu_ij's (submatrix column 2 ->end), transition event is numType -> numEvent-1
            x[numType] += -1
            x[numEvent-1] += 1
        
    return x # return population vector

def main(myStr):
    print('\n','Generating artificial data via birth-death process...')
    
    print('\n',' - Number of replicates: ', pin.R)
    print('\n',' - Time point(s): ', pin.T)
    print('\n',' - Initial condition:','\n', pin.X0,'\n')
    tic = time.perf_counter()
    data = dataGeneration(pin.paramsTrue,pin.X0,pin.T,pin.R)

    toc = time.perf_counter()

    np.save(myStr,data)

    print("Data saved to ",myStr,".npy",'\n',sep='')
    print(f"Run time: {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    myFileName = "myDataMLE"
    if len(sys.argv) > 1: # optional arguments passed in: file name
        myFileName = sys.argv[1]
    main("../data/"+myFileName)