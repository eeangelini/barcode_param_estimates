import numpy as np
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
from scipy.linalg import expm

from bgCodeMLENum import * # import functions from bgCodeMLE.py file
from ciMLENum import * # import functions for computing confidence intervals
from paramsInputNum import * # import parameters and options

import sys,time

def main(MLEDataString = "../data/mleOutput.npy"):
    # load data
    if withBarcodes:
        dataBarcodes = np.load("../data/myDataMLE.npy")
        
        if poolBCs:
            dataNum = np.sum(dataBarcodes,axis=0)[None,:,:,:] # sum over barcode lineages to just get numbers of each cell type, "dummy" dimension (only one initial condition)
            theta0init = simpleEstimateBarcodes(dataBarcodes,X0bc,T)  # "naive" parameter estimate
        else:
            dataNum = dataBarcodes # distinct "initial conditions" = distinct barcode lineages with corresponding initial numbers
            theta0init = simpleEstimateBarcodes(dataBarcodes,X0,T)  # "naive" parameter estimate
    else:
    # for recreation of Figure 10 from Gunnarsson et al:
        dataNum = np.zeros((I,K,L,R))
        dataNum[0,0,:,0] = np.array([1340,1746,2161,3259,4017,5584])
        dataNum[0,1,:,0] = np.array([21,75,181,393,681,1426])
        dataNum[1,0,:,0] = np.array([48,170,381,669,1357,2221])
        dataNum[1,1,:,0] = np.array([1533,2494,3896,6705,11431,16580])
        dataNum[0,0,:,1] = np.array([1313,1684,2184,3101,3898,5505])
        dataNum[0,1,:,1] = np.array([25,75,190,368,606,920])
        dataNum[1,0,:,1] = np.array([84,195,387,667,1387,2211])
        dataNum[1,1,:,1] = np.array([1628,2530,3906,6170,10566,15605])
        dataNum[0,0,:,2] = np.array([1276,1782,2242,3211,4224,5354])
        dataNum[0,1,:,2] = np.array([26,79,225,397,769,1021])
        dataNum[1,0,:,2] = np.array([60,164,381,651,1337,2459])
        dataNum[1,1,:,2] = np.array([1628,2564,4115,6275,10517,15998])

        theta0init = simpleEstimate(dataNum,X0,T)  # "naive" parameter estimate

    theta0, scale0 = simpleEstimateScaled(theta0init,lowerBounds,upperBounds,X0,T,deadCells) # scaled (and updated based on dead cell data, if using)
    if printInit:
        print("Initial guess:",'\n',theta0)
        print("Initial scale:", scale0)
    # pdb.set_trace() # breakpoint for debugging
    # First solve of the MLE problem given initial guess theta0
    print("Starting max likelihood step...")
    thetaMLE, scaleMLE, optLL, feasibleMLE = paramsMLE(dataNum,C,X0,T,theta0,scale0,Aineq,bineq,Aeq,beq,lowerBounds,upperBounds,tol,deadCells,printBounds)
    print("End max likelihood step")

    # Following two blocks are options to re-solve MLE problem under various inputs.
    # If log likelihood value is better upon re-solving, we choose that solution over the previous one.
    # First option is to re-solve via user-supplied initial guess theta0def.
    if len(theta0def) > 0: # if theta0def is NOT empty
        scale0def = 10**np.floor(np.log10(abs(theta0def[:,:,n])))
        scale0def = np.where(scale0def == 0,1,scale0def) # replace zero entries with 1, leave rest
        for n in range(theta0def.shape[-1]): # loop through third dimension of tensor theta0def - number of initial guesses supplied
            thetaTemp, scaleTemp, optLLTemp, feasibleTemp = paramsMLE(dataNum,C,X0,T,theta0def[:,:,n],scale0def,Aineq,bineq,Aeq,beq,lowerBounds,upperBounds,tol,deadCells)       
            
            if optLLTemp < optLL: # if we get a "better" estimate (i.e., minimize neg log likelihood further), update
                thetaMLE = thetaTemp
                scaleMLE = scaleTemp
                optLL = optLLTemp
                feasibleMLE = feasibleTemp

    # Second option involves solving mutlitple times using a) simple estimate as initial guess or b) a random initial guess.
    if nOptSimple > 0 or nOptRandom > 0:
        for n in range(nOptSimple):
            theta0, scale0 = simpleEstimateScaled(X0,T,theta0init,lowerBounds,upperBounds,deadCells)
            thetaTemp, scaleTemp, optLLTemp, feasibleTemp = paramsMLE(dataNum,C,X0,T,theta0,scale0,Aineq,bineq,Aeq,beq,lowerBounds,upperBounds,tol,deadCells)

            if optLLTemp < optLL: # if we get a "better" estimate (i.e., minimize neg log likelihood further), update
                thetaMLE = thetaTemp
                scaleMLE = scaleTemp
                optLL = optLLTemp
                feasibleMLE = feasibleTemp

        for n in range(nOptRandom):
            theta0, scale0 = simpleEstimateScaled(X0,T,[],lowerBounds,upperBounds,deadCells)
            thetaTemp, scaleTemp, optLLTemp, feasibleTemp = paramsMLE(dataNum,C,X0,T,theta0,scale0,Aineq,bineq,Aeq,beq,lowerBounds,upperBounds,tol,deadCells)

            if optLLTemp < optLL: # if we get a "better" estimate (i.e., minimize neg log likelihood further), update
                thetaMLE = thetaTemp
                scaleMLE = scaleTemp
                optLL = optLLTemp
                feasibleMLE = feasibleTemp

    print("MLE parameter estimates: ",'\n',thetaMLE)
    # pdb.set_trace() # breakpoint for debugging

    # Get confidence intervals
    print("Getting MLE confidence intervals...",'\n')
    tic = time.perf_counter()
    ciMLE, ci_yvalues_num, ci_check_num = getCI(alpha_q,ciOption,nOptCI,dataNum,deadCells,C,X0,T,thetaMLE,scaleMLE,optLikelihood,Aineq,bineq,Aeq,beq,lowerBounds,upperBounds,tol)
    toc = time.perf_counter()
    print(f"MLE CIs ({100*(1-alpha_q):0.2f} %)",'\n',ciMLE,'\n')
    print(f"- Time to obtain: {toc - tic:0.4f} seconds")


    # Save parameter estimates and confidence intervals for bar plots (ciPlot.py)
    # parameters, CI lower bounds, and CI upper bounds are rows of outputMLE
    # flatten using column-major convention (for plotting)
    outputMLE = np.vstack((thetaMLE.flatten('F'),ciMLE[0,:,:].flatten('F'),ciMLE[1,:,:].flatten('F')))
    np.save(MLEDataString,outputMLE)

    ## Data visualization
    # If dataVis is set to True, plots are produced that show for each
    # initial condition and each type how well the statistical model fits the
    # data. More precisely, the mean prediction of the statistical model and
    # 1-alpha_q confidence intervals, under the assumption that the MLE
    # estimates are the true parameters, are compared with the data.
    if dataVis:
        quant_norm = norm.ppf(1-alpha_q/2)
        b,A = extractParams(thetaMLE,[])
        for i in range(I):
            for j in range(C[i].shape[1]):
                plt.figure()
                c_j = C[i][:,j]
                n = L*10
                tVec = np.linspace(0,T[-1],n)
                f_mean = np.zeros(n)
                f_lb = np.zeros(n)
                f_ub = np.zeros(n)
                for l in range(n):
                    f_mean[l] = X0[i,:].dot(expm(tVec[l]*A).dot(c_j))
                    f_lb[l] = f_mean[l]-quant_norm*np.sqrt(c_j.dot(covMat(X0[i,:],A,b,tVec[l],[])).dot(c_j))
                    f_ub[l] = f_mean[l]+quant_norm*np.sqrt(c_j.dot(covMat(X0[i,:],A,b,tVec[l],[])).dot(c_j))

                plt.plot(tVec,f_lb)
                plt.plot(tVec,f_ub)
                plt.fill_between(tVec,f_lb,f_ub,interpolate=True,color='#5f5f5f') # shade inbetween upper and lower bounds
                plt.plot(tVec,f_mean,'k')
                y = np.zeros(L)
                for r in range(R):
                    for l in range(L):
                        y[l] = dataNum[i,:,l,r].dot(c_j)
                    plt.scatter(np.concatenate((np.zeros(1),T)), np.hstack((X0[i,:].dot(c_j), y)), s=200, c='k',marker='x',linewidths=1)
        plt.show()


if __name__ == '__main__':
    dataFileName = "mleOutput"
    if len(sys.argv) == 2: # optional argument passed in: file name extension (underscore) for saving output data (MLE + CIs)
        dataFileName = dataFileName + "_" + sys.argv[1]
    elif len(sys.argv) > 2: # optional arguments passed in: unique file name for both saving output data (indicated by "u")
        dataFileName = sys.argv[1]
    main("../data/"+dataFileName+".npy")