import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dataGen import *
from cellNumberBarcode import *
import paramsInput as pin
import sys

global numReps, initFrac, bcFids
numReps = [10,50,100,500,1000] # list of number of replicates (aka number of data points)
initFrac = [0.9,0.7,0.5,0.3,0.1]
bcFids = [1.0,0.8,0.6,0.4,0.2,0.01]

def getError(myCond, myStr=""):
    if myCond == "R":
        absError = np.zeros((pin.K,pin.K,len(numReps)))
        relError = np.zeros((pin.K,pin.K,len(numReps)))

        for (jj,R) in enumerate(numReps):
            print("Number of replicates: ",R,'\n')
            data = dataGeneration(pin.paramsTrue,pin.X0,pin.T,R)
            print("Run complete",'\n')

            avgLineages = data.mean(axis = 3) # take mean over replicates
            rateMatList = approxRates(pin.T,avgLineages,pin.X0) # get list of estimates of A (KxK matrices x L-1 time points)
            rateMatEst = sum(rateMatList)/pin.L

            absError[:,:,jj] = abs(rateMatEst-pin.ATrue)
            relError[:,:,jj] = np.divide(abs(rateMatEst-pin.ATrue),abs(pin.ATrue))

    elif myCond == "F0":
        absError = np.zeros((pin.K,pin.K,len(initFrac)))
        relError = np.zeros((pin.K,pin.K,len(initFrac)))

        for (jj,F0) in enumerate(initFrac):
            print("Initial fraction of type 1 cells: ",F0,'\n')
            N0 = 2000
            X0 = np.zeros((pin.K,pin.K))
            X0[0,0] = np.floor(F0*N0)
            X0[1,1] = N0 - X0[0,0]
            data = dataGeneration(pin.paramsTrue,X0,pin.T,pin.R)
            print("Run complete",'\n')

            avgLineages = data.mean(axis = 3) # take mean over replicates
            rateMatList = approxRates(pin.T,avgLineages,X0) # get list of estimates of A (KxK matrices x L-1 time points)
            rateMatEst = sum(rateMatList)/pin.L

            absError[:,:,jj] = abs(rateMatEst-pin.ATrue)
            relError[:,:,jj] = np.divide(abs(rateMatEst-pin.ATrue),abs(pin.ATrue))
    
    elif myCond == "rho":
        absError = np.zeros((pin.K,pin.K,len(bcFids),len(bcFids)))
        relError = np.zeros((pin.K,pin.K,len(bcFids),len(bcFids)))

        for (ii,rho1) in enumerate(bcFids):
            for (jj,rho2) in enumerate(bcFids):
                print("Barcode 1 fidelity: ",rho1,'\n')
                print("Barcode 2 fidelity: ",rho2,'\n')
                print("Total barcode fidelity: ",rho1*rho2,'\n')
                X0 = np.zeros((pin.K,pin.K))
                X0[0,:] = np.array([rho1,1-rho1])*1000
                X0[1,:] = np.array([1-rho2,rho2])*1000
                data = dataGenerationEA(pin.paramsTrue,X0,pin.T,pin.R)
                print("Run complete",'\n')

                avgLineages = data.mean(axis = 3) # take mean over replicates
                rateMatList = approxRates(pin.T,avgLineages,X0) # get list of estimates of A (KxK matrices x L-1 time points)
                rateMatEst = sum(rateMatList)/pin.L

                absError[:,:,ii,jj] = abs(rateMatEst-pin.ATrue)
                relError[:,:,ii,jj] = np.divide(abs(rateMatEst-pin.ATrue),abs(pin.ATrue))

    np.save("../data/absError_change"+myCond+myStr,absError)
    np.save("../data/relError_change"+myCond+myStr,relError)

    return

def plotError(myCond):
    absError = []
    relError = []

    
    if myCond == "R":
        numRuns = 16 # TBD by user, number of independent runs of getError
        for l in range(1,numRuns):
            absError.append(np.load("../data/absError_changeR"+str(l)+".npy"))
            relError.append(np.load("../data/relError_changeR"+str(l)+".npy"))
        absError = sum(absError)/numRuns
        relError = sum(relError)/numRuns

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))

        ax1.loglog(numReps,absError[0,0,:],'b.-',linewidth=3,alpha = 0.8,label='$g_1$')
        ax1.loglog(numReps,absError[1,1,:],'b.--',linewidth=3,alpha = 0.8,label='$g_2$')
        ax1.loglog(numReps,absError[1,0,:],'k.-',linewidth=3,label='$k_{12}$')
        ax1.loglog(numReps,absError[0,1,:],'k.--',linewidth=3,label='$k_{21}$')
        ax1.set_xlabel("Number of data points",fontsize=15)
        ax1.set_ylabel("Absolute error of estimation",fontsize=15)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.legend(title="Parameter:",loc="upper right")

        ax2.loglog(numReps,relError[0,0,:],'b.-',linewidth=3,alpha = 0.8,label='$g_1$')
        ax2.loglog(numReps,relError[1,1,:],'b.--',linewidth=3,alpha = 0.8,label='$g_2$')
        ax2.loglog(numReps,relError[1,0,:],'k.-',linewidth=3,label='$k_{12}$')
        ax2.loglog(numReps,relError[0,1,:],'k.--',linewidth=3,label='$k_{21}$')
        ax2.set_xlabel("Number of data points",fontsize=15)
        ax2.set_ylabel("Relative error of estimation",fontsize=15)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.legend(title="Parameter:",loc="lower left")
        
        plt.setp((ax1,ax2), ylim=[ax1.get_ylim()[0],ax2.get_ylim()[1]])
        fig.suptitle("Error of parameter estimates vs number of data points; mean of "+str(numRuns)+ " replicates", fontsize=15)
        fig.tight_layout(pad=5.0)
        plt.subplots_adjust(top=0.89)
    elif myCond == "F0":
        numRuns = 5
        for l in range(1,numRuns):
            absError.append(np.load("../data/absError_changeF0"+str(l)+".npy"))
            relError.append(np.load("../data/relError_changeF0"+str(l)+".npy"))
        absError = sum(absError)/numRuns
        relError = sum(relError)/numRuns

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))

        ax1.semilogy(initFrac,absError[0,0,:],'b.-',linewidth=3,alpha = 0.8,label='$g_1$')
        ax1.semilogy(initFrac,absError[1,1,:],'b.--',linewidth=3,alpha = 0.8,label='$g_2$')
        ax1.semilogy(initFrac,absError[1,0,:],'k.-',linewidth=3,label='$k_{12}$')
        ax1.semilogy(initFrac,absError[0,1,:],'k.--',linewidth=3,label='$k_{21}$')
        ax1.set_xlabel("Initial fraction of type 1 cells",fontsize=15)
        ax1.set_ylabel("Absolute error of estimation",fontsize=15)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.legend(title="Parameter:",loc="upper left")

        ax2.semilogy(initFrac,relError[0,0,:],'b.-',linewidth=3,alpha = 0.8,label='$g_1$')
        ax2.semilogy(initFrac,relError[1,1,:],'b.--',linewidth=3,alpha = 0.8,label='$g_2$')
        ax2.semilogy(initFrac,relError[1,0,:],'k.-',linewidth=3,label='$k_{12}$')
        ax2.semilogy(initFrac,relError[0,1,:],'k.--',linewidth=3,label='$k_{21}$')
        ax2.set_xlabel("Initial fraction of type 1 cells",fontsize=15)
        ax2.set_ylabel("Relative error of estimation",fontsize=15)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.legend(title="Parameter:",loc="upper left")

        plt.setp((ax1,ax2), ylim=[ax1.get_ylim()[0],ax2.get_ylim()[1]])
        fig.suptitle("Error of parameter estimates vs initial fraction of type 1 cells; mean of "+str(pin.R)+" data points x "+str(numRuns)+ " replicates", fontsize=15)
        fig.tight_layout(pad=5.0)
        plt.subplots_adjust(top=0.89)

    elif myCond == "rho":
        numRuns = 4
        for l in range(1,numRuns):
            absError.append(np.load("../data/absError_changerho"+str(l)+".npy"))
            relError.append(np.load("../data/relError_changerho"+str(l)+".npy"))
        absError = sum(absError)/numRuns
        relError = sum(relError)/numRuns

        fig, ax = plt.subplots(4, 2,figsize=(12,24),sharey='col')

        rc = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='$\\rho_1 = \\rho_2 = 1$')
        bt = mlines.Line2D([], [], color='blue', marker='^', linestyle='None', label='$\\rho_1 = 1$')
        bs = mlines.Line2D([], [], color='blue', marker='s', linestyle='None', label='$\\rho_2= 1$')
        bd = mlines.Line2D([], [], color='black', marker='.', linestyle='None', label='other')

        myLabels = ["$g_1$","$g_2$","$k_{12}$","$k_{21}$"]
        myIdx = [(0,0),(1,1),(1,0),(0,1)]

        for i in range(pin.K**2):
            rho = np.outer(bcFids,bcFids).flatten()
            for (ii,rho1) in enumerate(bcFids):
                for (jj,rho2) in enumerate(bcFids):
                    if rho1 == 1:
                        if rho2 == 1:
                            myStyle = 'ro'
                        else:
                            myStyle = 'b^'
                    elif rho2 == 1:
                        myStyle = 'bs'
                    else:
                        myStyle = 'k.'
                    # Different ways of defining "overall fidelity"
                    rho = 0.5*(rho1+rho2)
                    # rho = np.sqrt(rho1*rho2)
                    # rho = min(rho1,rho2)
                    # rho = max(rho1,rho2)
                    # rho = rho1*rho2
                    # rho = rho1/rho2

                    ax[i,0].semilogy(rho,absError[myIdx[i][0],myIdx[i][1],ii,jj].flatten(),myStyle,ms=8)
                    ax[i,0].tick_params(axis='x', labelsize=14)
                    ax[i,0].tick_params(axis='y', labelsize=14)
                    ax[i,0].set_xlabel("Barcode fidelity $\\rho = \\frac{\\rho_1 + \\rho_2}{2}$",fontsize=15)
                    ax[i,0].set_ylabel("Absolute error of estimation",fontsize=15)
                    ax[i,0].set_title("Estimate of "+myLabels[i])
                    ax[i,0].legend(handles=[rc,bt,bs,bd],loc="lower left")

                    ax[i,1].semilogy(rho,relError[myIdx[i][0],myIdx[i][1],ii,jj].flatten(),myStyle,ms=8)
                    ax[i,1].tick_params(axis='x', labelsize=14)
                    ax[i,1].tick_params(axis='y', labelsize=14)
                    ax[i,1].set_xlabel("Barcode fidelity $\\rho = \\frac{\\rho_1 + \\rho_2}{2}$",fontsize=15)
                    ax[i,1].set_ylabel("Relative error of estimation",fontsize=15)
                    ax[i,1].set_title("Estimate of "+myLabels[i])
                    ax[i,1].legend(handles=[rc,bt,bs,bd],loc="lower left")
        
        # for full colormap plots
        # fig, ax = plt.subplots(4, 2,figsize=(12,24))

        # myLabels = ["$g_1$","$g_2$","$k_{12}$","$k_{21}$"]
        # myIdx = [(0,0),(1,1),(1,0),(0,1)]

        # for i in range(pin.K**2):
        #     plot1 = ax[i,0].imshow(absError[myIdx[i][0],myIdx[i][1],:,:],origin='lower')
        #     ax[i,0].tick_params(axis='x', labelsize=14)
        #     ax[i,0].tick_params(axis='y', labelsize=14)
        #     ax[i,0].set_xlabel("Barcode 1 fidelity $\\rho_1$",fontsize=15)
        #     ax[i,0].set_ylabel("Barcode 2 fidelity $\\rho_2$",fontsize=15)
        #     ax[i,0].set_xticks(np.arange(len(bcFids)),labels=map(str,bcFids))
        #     ax[i,0].set_yticks(np.arange(len(bcFids)),labels=map(str,bcFids))
        #     ax[i,0].set_title("Absolute error of estimate for "+myLabels[i])
        #     plt.colorbar(plot1,ax=ax[i,0],fraction=0.046, pad=0.04)

        #     plot2 = ax[i,1].imshow(relError[myIdx[i][0],myIdx[i][1],:,:],origin='lower')
        #     ax[i,1].tick_params(axis='x', labelsize=14)
        #     ax[i,1].tick_params(axis='y', labelsize=14)
        #     ax[i,1].set_xlabel("Barcode 1 fidelity $\\rho_1$",fontsize=15)
        #     ax[i,1].set_ylabel("Barcode 2 fidelity $\\rho_2$",fontsize=15)
        #     ax[i,1].set_xticks(np.arange(len(bcFids)),labels=map(str,bcFids))
        #     ax[i,1].set_yticks(np.arange(len(bcFids)),labels=map(str,bcFids))
        #     ax[i,1].set_title("Relative error of estimate for "+myLabels[i])
        #     plt.colorbar(plot2,ax=ax[i,1],fraction=0.046, pad=0.04)
        
        plt.setp(ax, ylim=[ax[0,0].get_ylim()[0],ax[0,1].get_ylim()[1]])
        fig.suptitle("Error of parameter estimates vs barcode fidelity; mean of "+str(pin.R)+" data points x "+str(numRuns)+ " replicates", fontsize=15)
        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95)
    

    plt.savefig("../figs/error_change"+myCond+".png",dpi=500)
    plt.show()

    return

if __name__ == '__main__':
    if sys.argv[1] == "plot": # plot error for changing sys.argv[2]
        plotError(sys.argv[2])
    else: # get error, first error is condition that you are changing (number of data points, initial conditions, etc)
        if len(sys.argv) > 2:
            getError(sys.argv[1],myStr=sys.argv[2]) # optional argument is name of string for saving file
        else:
            getError(sys.argv[1])