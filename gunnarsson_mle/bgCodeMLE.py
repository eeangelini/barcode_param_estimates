import numpy as np
from scipy.linalg import logm, expm
import scipy.optimize as opt
import scipy.integrate as scint
from paramsGen import generateParams, extractParams # for generating parameters and getting infin. generator & vector of birth rates

# Initialize random number generator, random seed can be fixed for reproducibility
rndseed = 1
global rng
rng = np.random.default_rng(rndseed)

def simpleEstimateBarcodes(dataBC,X0,T):
    ''' Simple parameter estimate via inverting mean behavior using barcode lineage data '''
    L = len(T)
    K = dataBC.shape[1]

    meanData = np.mean(dataBC,axis=3) # take mean over replicates

    AHat = np.zeros((K,K)) # estimate of A for each estimate of R(t) = e^(tA): (1/t)*logm(R(t)), average over t and replicates

    for l in range(L):
        Rmat = np.zeros((K,K)) 
        for i in range(K):
            for j in range(K):
                # R_ij(t) ~= lineage j cells in type i at t / initial number of lineage j cells
                Rmat[i,j] = meanData[i,j,l]/X0[j,:].sum()
        AHat += (1/T[l])*logm(Rmat.T) # code deals with A as defined as transpose of how we define it
    
    AHat = AHat/L

    # turn into corresponding estimates of parameters theta (KxK+1 array)
    theta0 = np.zeros((K,K+1))
    theta0[:,1] = AHat.diagonal() + np.sum(AHat-np.diag(AHat.diagonal()),axis=1) # net growth rates
    # estimate nu_ij's
    for i in range(K):
        theta0[i,2:K+1] = AHat[i,list(range(i))+list(range(i+1,K))] # transition rates

    return theta0

def simpleEstimate(data,X0,T):
    ''' Simple parameter estimate via inverting mean behavior '''
    K = data.shape[1]
    L = data.shape[2]
    R = data.shape[3]

    # simple parameter estimate to use as initial condition for MLE solver
    theta0 = np.zeros((K,K+1))

    v = 0
    invX0 = np.linalg.inv(X0.T@X0) # note: this doesn't work for only one starting condition
    for l in range(L):
        for r in range(R):
            X = data[:,:,l,r]
            myEigsReal = np.linalg.eig(invX0@X0.T@X)[0][np.isreal(np.linalg.eig(invX0@X0.T@X)[0])]
            if all(myEigsReal >= 0): # if no negative real eigenvalues, matrix log is unique
                AHat = logm(invX0@X0.T@X)/T[l]
                # estimate lambda_i's
                theta0[:,1] = theta0[:,1] + AHat.diagonal() + np.sum(AHat-np.diag(AHat.diagonal()),axis=1)
                # estimate nu_ij's
                for i in range(K):
                    theta0[i,2:K+1] = theta0[i,2:K+1] + AHat[i,list(range(i))+list(range(i+1,K))]
                v += 1
    theta0 = theta0/v

    return theta0

def simpleEstimateScaled(theta0in,lb,ub,X0,T,deadCells = []):
    ''' First SCALED estimate of model parameters (either using number of dead cells
    an optional argument, or using estimate of theta from simpleEstimate, theta0in) '''
    I = X0.shape[0] # number of initial conditions supplied
    K = X0.shape[1] # number of cell types

    if len(theta0in) > 0:
        theta0 = theta0in # initial parameter vector guess theta0
        if len(deadCells) > 0:
            D = np.zeros((I,K))
            c = np.zeros((I,1))
            for i in range(I):
                for j in range(K):
                    D[i,j] = X0[i,j]*(1/theta0in[j,1])*(np.exp(theta0in[j,1]*T[0])-1)

                c[i] = np.mean(deadCells[i,0,0,:])

            for j in range(K):
                theta0[:,0] = theta0in[:,1] + (np.linalg.inv(D.T@D)@D.T).dot(c)

        else: # estimate b_i by |lambda_i|/U(0,0.75), if above ub or below lb pick the bounds
            bEst = np.divide(abs(theta0[:,1]),rng.uniform(0,0.75,K))
            theta0[:,0] = np.maximum(np.minimum(bEst,ub[:,0]),lb[:,0])
 
    else:
        theta0 = generateParams(K,1)[1] # else, randomly generate parameters

    if abs(theta0 == 0).any():
        theta0 = np.where(theta0==0,1,theta0) # replace 0s with 1s so that scale = 1 where x0 = 0
    
    scale0 = 10**np.floor(np.log10(abs(theta0)))
    scale0 = np.where(scale0 == 0, 1, scale0)

    return theta0, scale0

def paramsMLE(data,C,X0,T,theta0,scale0,Aineq,bineq,Aeq,beq,lb,ub,tol,deadCells,printBounds = False):
    ''' MLE estimation of model parameters'''

    Aineq,bineq,Aeq,beq,lb,ub = scaleConstraints(Aineq,bineq,Aeq,beq,lb,ub,scale0)
    theta0 = np.divide(theta0,scale0) # initial parameter guess for MLE optimization step

    K = theta0.shape[0]
    theta0 = theta0.flatten() # must flatten to pass into opt.minimize
    scale0 = scale0.flatten()
    lb = lb.flatten()
    ub = ub.flatten()

    if printBounds:
        print("Scaled inequality constraints:",'\n',Aineq,bineq)
        print("Scaled equality constraints:",'\n',Aeq,beq)
        print("Scaled lower bounds: ",lb)
        print("Scaled upper bounds: ",ub)
        print("Scaled initial guess: ",theta0)

    myconstr = []
    if len(Aeq) > 0:
        Aeq = Aeq.reshape((len(beq),K*(K+1)))
        lin_eq = opt.LinearConstraint(Aeq,beq,beq)
        myconstr.append(lin_eq)

    if len(Aineq) > 0:
        Aineq = Aineq.reshape((len(bineq),K*(K+1)))
        lin_ineq = opt.LinearConstraint(Aineq,ub=bineq)
        myconstr.append(lin_ineq)
    
    res = opt.minimize(lambda x: negLL(x,scale0,C,X0,data,T,deadCells),
               theta0, method='SLSQP',
               constraints=myconstr, 
               options={'ftol':1e-10,'disp': True},
               bounds=opt.Bounds(lb,ub))

    thetaMLE = res.x # resulting parameters from MLE step
    optValue = res.fun # value of objective function (negative log likelihood) at optimal parameters

    # Check if resulting optimal theta satisfies the given constraints (inequality and equality)
    # i.e., is an element of the set of "feasible" parameters
    feasible = 0
    feasible = feasible + sum(Aineq.dot(thetaMLE) - bineq > tol)
    if len(Aeq) > 0:
        feasible = feasible + sum(abs(Aeq.dot(thetaMLE)) - beq > tol)
    feasible = feasible + sum(lb > thetaMLE) + sum(thetaMLE > ub)
    # from above, if ALL parameters lie within the constraints, feasible == 0  
    if feasible == 0:
        feasible = 1 # so set equal to 1 (i.e., True)
    else: # else, at least one parameter value does not satisfy constraints within tol
        feasible = 0 # so set equal to 0 (i.e., False)

    thetaMLE = thetaMLE*scale0 # rescale
    scaleMLE = 10**np.floor(np.log10(abs(thetaMLE)))
    scaleMLE = np.where(scaleMLE == 0, 1, scaleMLE)

    return thetaMLE.reshape((K,K+1)), scaleMLE.reshape((K,K+1)), optValue, feasible

def scaleConstraints(Aineq,bineq,Aeq,beq,lb,ub,scale):
    '''Imposes inequality constraint lambda <= b, upper
    and lower bounds for parameters, and and any user input 
    constraints (Aineq/bineq, Aeq/beq). Then scales these 
    constraints by input scale.'''
    K = scale.shape[0]

    # build full matrix of inequality constraints
    AineqDef = np.zeros((2,K*(K+1)))
    for i in range(K): # lambda <= b
        AineqDef[i,i*(K+1)] = -1
        AineqDef[i,i*(K+1)+1] = 1
    bineqDef = np.zeros(K)
    # augment with user supplied constraints if provided
    if len(Aineq) > 0:
        for n in range(Aineq.shape[0]):
            AineqDef = np.vstack((AineqDef, Aineq[n,:]))
            bineqDef = np.concatenate(bineqDef, bineq[n])

    # Rescale by scale
    AineqScaled = AineqDef
    for j in range(AineqScaled.shape[0]):
        AineqScaled[j,:] = AineqScaled[j,:]*scale.flatten()
    bineqScaled = bineqDef

    # equality constraints, if provided
    AeqDef = []
    beqDef = []
    if len(Aeq) > 0:
        for n in range(Aeq.shape[0]):
            AeqDef = np.vstack((AeqDef, Aeq[n,:]))
            beqDef = np.concatenate(beqDef, beq[n])
        for j in range(AeqDef.shape[0]):
            AineqDef[j,:] = AineqDef[j,:]*scale.flatten()

    AeqScaled = AeqDef
    beqScaled = beqDef

    # Parameter upper and lower bounds
    lbDef = np.hstack((np.zeros((K,1)),-np.inf*np.ones((K,1)),np.zeros((K,K-1))))
    if len(lb) == 0:
        lbScaled = np.divide(lbDef,scale)
    else:
        lbScaled = np.divide(np.maximum(lbDef,lb),scale)

    ubScaled = np.divide(ub,scale)

    return AineqScaled,bineqScaled,AeqScaled,beqScaled,lbScaled,ubScaled

def negLL(theta,scale,C,X0,data,T,deadCells):
    ''' Negative log likelihood function '''

    theta = theta*scale

    I = data.shape[0] # number of initial conditions
    K = data.shape[1] # number of cell types
    L = data.shape[2] # number of time points
    R = data.shape[3] # number of replicates

    b,A = extractParams(np.reshape(theta,(K,K+1)),deadCells) # get birth rates b, generator matrix A from matrix params
    if len(deadCells) > 0:
        for i in range(I):
            C[i] = np.vstack(np.hstack(C[i],np.zeros(C[i].shape[0])),np.hstack(np.zeros(C[i].shape[1]),1))
        X0 = np.hstack(X0,np.zeros(X0.shape[0]))

    value = 0
    for i in range(I):
        for l in range(L):
            meanVec = X0[i,:]@expm(T[l]*A)
            SM = C[i].T@covMat(X0[i,:],A,b,T[l],deadCells)@C[i]
            if np.linalg.det(SM) == 0:
                print('\n',"***ERROR***",'\n',"Singular covariance matrix",'\n')
                print("Covariance matrix:",'\n',SM)
                print("Generator matrix:", '\n', A)
                print("Mean vector: ",meanVec)
                print("Initial conditions: ",X0[i,:])
                print("Time point: ",T[l])
                print('\n')
                exit()
            invSM = np.linalg.inv(SM)
            for r in range(R):
                if len(deadCells)>0:
                    dMinusMu = (np.hstack((data[i,:,l,r],deadCells[i,0,l,r])) - meanVec)@C[i]
                else:
                    dMinusMu = (data[i,:,l,r] - meanVec)@C[i]
                value += dMinusMu@invSM@dMinusMu.T
            value += R*np.log(np.linalg.det(SM))
    #print(param, '\n',value)
    return value

def covMat(x0,A,b,t,deadCells = []):
    ''' Covariance matrix at time point t of birth-death process with initial 
    state x0, generator A, pure birth rates b. Optional input deadCells gives 
    number of dead cells at time t.
    
    Expression from Gunnarsson et al, Appendix C (note - this gives covariance matrix 
    saled by initial number of cells - statistical model of cell NUMBER data as opposed
    to cell fraction data)
    '''
    K = A.shape[0]
    SM = np.zeros((K,K))

    if len(deadCells) > 0:
        b = np.hstack(b,0)

    for j in range(K):
        if x0[j]>0:
            meanVec = expm(t*A)[j,:]
            f = lambda s: 2*expm((t-s)*A).T@np.diag(b*(expm(s*A)[j,:]))@expm((t-s)*A)
            SM = SM + x0[j]*(scint.quad_vec(f,0,t)[0]+np.diag(meanVec)-np.outer(meanVec,meanVec))

    return SM



