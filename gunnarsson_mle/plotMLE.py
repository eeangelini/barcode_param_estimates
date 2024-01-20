import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from paramsInputMLE import paramsTrue

import sys

def main(dataString,plotString):
    ''' Plots max likelihood parameter estimations along with confidence intervals 
    as a horizontal bar plot using output from cellNumberMLE.py. '''
    # plot MLE + CI, true values of params
    # flattened arrays using column-major convention: (b_1,b_2,...,lam_1,lam_2,...,v_12,v_21,...)
    thetaTrue = paramsTrue.flatten('F')
    outputMLE = np.load(dataString)
    thetaMLE = outputMLE[0,:]
    CI = outputMLE[1:3,:]

    # uncomment to recreate left panel of Figure 10 from Gunnarsson et al (2023)
    # thetaMLE = np.array([[0.46255841, 0.2972822, 0.01980745],
    #                      [0.67045032, 0.50500118, 0.03965353]]).flatten('F')
    # lbCI = np.array([[0.32429318, 0.29217245, 0.01840231],
    #                  [0.5036399, 0.49993471, 0.03832886]]).flatten('F')
    # ubCI = np.array([[0.75834511, 0.30227053, 0.02127927],
    #                  [1.01907149, 0.51008724, 0.04103335]]).flatten('F')
    # CI = np.vstack((lbCI,ubCI))

    myIter = np.arange(len(thetaMLE)) # to iterate over each parameter
    delta = 0.2 # wdith of error boxes in y direction

    fig,ax = plt.subplots(figsize = (7,6.5)) # set figure size

    # plot MLE with "error bars" (vertical line, length of box) at point of estimate
    ax.errorbar(thetaMLE, myIter,yerr=delta*np.ones(len(thetaMLE)),fmt='None',ecolor='k',elinewidth=0.75)

    # Create list for all the error patches - for plotting confidence interval boxes
    errorBoxes = []
    # Loop over data points; create box from CIs at each point
    for i in myIter:
        rect = Rectangle((CI[0,i],i-delta),CI[1,i]-CI[0,i],2*delta)
        errorBoxes.append(rect)
    # Create patch collection with specified color/alpha
    pc = PatchCollection(errorBoxes,facecolor=(0,0,1,0.2),edgecolor=(0,0,0,1.0),linewidth=0.75)
    # Add collection to axes (i.e., show)
    ax.add_collection(pc)

    # Plot arrows at true parameter values
    for i in myIter:
        plt.plot(thetaTrue[i],i+1.4*delta, 'kv', markersize=6)
        plt.plot([thetaTrue[i],thetaTrue[i]],[i+1.6*delta,i+3.15*delta], 
                'k-', linewidth=1.5)

    # Add labels for each parameter
    ax.set_yticks(ticks=myIter,labels=["$b_1$","$b_2$",
                    "$\lambda_1$","$\lambda_2$",
                    "$\\nu_{12}$","$\\nu_{21}$"])
    ax.yaxis.set_ticks_position("right") # Put parameter name labels on the right
    ax.tick_params(axis='y',which='both',length=0, labelsize=14) # Hide tick lines
    ax.tick_params(axis='x',which='both',direction="in",labelsize=14) # x axis ticks inside plot

    for direction in ["left", "right", "top"]: # Hide all axis lines except bottom
        ax.spines[direction].set_visible(False)

    ax.set_xlim([1e-2,1e2]) # set x lims
    ax.set_xscale("log") # log scale on x-axis

    # Put boxes around each row
    paramBoxes = []
    for i in myIter:
        rect = Rectangle((1e-2+0.0005,i-delta),(1e2-1e-2)-5,2*delta)
        paramBoxes.append(rect)
    pc = PatchCollection(paramBoxes,facecolor='None',edgecolor=(0,0,0,1.0),linewidth=0.75)
    # Add collection to axes (i.e., show)
    ax.add_collection(pc)

    # Pad margins
    ax.margins(0.11)

    # Set title
    ax.set_title("MLE Parameter Estimates (Cell Number Data)", fontsize=16,fontname="Times")

    # Save figure
    plt.savefig(plotString,dpi=500)

    # Show
    plt.show()

if __name__ == '__main__':
    dataFileName = "mleOutput"
    plotFileName = "mlePlot"
    if len(sys.argv) == 2: # optional arguments passed in: file name extension (underscore) for both loading data and saving plot
        dataFileName = dataFileName + "_" + sys.argv[1]
        plotFileName = plotFileName + "_" + sys.argv[1]
    elif len(sys.argv) > 2: # optional arguments passed in: unique file names for both loading data and saving plot
        if len(sys.argv[1]) > 0: # if string is not empty...
            dataFileName = sys.argv[1]
        if len(sys.argv[2]) > 0:
            plotFileName = sys.argv[2]
    main("../data/"+dataFileName+".npy","../figs/"+plotFileName+".png")