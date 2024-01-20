import numpy as np

# Initialize random number generator, random seed can be fixed for reproducibility
global rng
rng = np.random.default_rng()
# rndseed = 1 
# rng = np.random.default_rng(rndseed)

def generateParams(K,nParamRegimes):
    ''' 
    Input:
        - K: number of cell types
        - nParamRegimes: number of distinct parameter sets (aka regimes). 
    
    Output:
        - paramsTrue (K x (K+1) x nParamRegimes): nParamRegimes sets of K x (K+1) model parameters (to be estimated)
        - scaleTrue (K x (K+1) x nParamRegimes): scale (order of magnitude) of each parameter in paramsTrue
        - X0 (K x K x nParamRegimes): Initial conditions to be supplied for each parameter set
        - Ntot (1 x K x nParamRegimes): Vector of initial number of each cell type (from X0) for each parameter set
    '''
    paramsTrue = np.zeros((K,K+1,nParamRegimes)) # parameters
    scaleTrue = np.zeros((K,K+1,nParamRegimes)) # parameter scales

    X0 = np.zeros((K,K,nParamRegimes)) # initial conditions
    Ntot = np.zeros((1,K,nParamRegimes)) # total number of starting cells of each phenotype

    for reg in range(nParamRegimes):
        cond = 0
        while cond == 0:
            # Randomly sample parameters:
            # b,d, (lambda): Uniform(0,1) 
            # nu: log uniform between 10^-3 and 10^-1
            paramsTrue[:,:,reg] = np.hstack((rng.uniform(0,1,(K,2)),10**(-3+2*rng.uniform(0,1,(K,K-1))))) 
            paramsTrue[:,1,reg] = paramsTrue[:,0,reg] - paramsTrue[:,2,reg] # Net growth rate

            # Conditions for generating "good" parameter sets:
            # Do not have negative growth rate for every cell type 
            # and b, lambda are both greater than 0.01 in absolute 
            # value for every cell type
            if sum(paramsTrue[:,1,reg]<0) != K and sum(abs(paramsTrue[:,0:1,reg])<0.01) <= 0:
                cond = 1 # break while loop
                # get paramter scales
                scaleTrue = 10**np.floor(np.log10(abs(paramsTrue)))
                scaleTrue = np.where(scaleTrue == 0, 1, scaleTrue)
        
        # Generate initial conditions based on 
        # order of magnitude of switching rates
        if np.log10(paramsTrue[:,:,reg]).min() >= -2:
            X0[:,:,reg] = 10**3*np.eye(K)
        elif np.log10(paramsTrue[:,:,reg]).min() >= -3:
            X0[:,:,reg] = 10**4*np.eye(K)
        else:
            X0[:,:,reg] = 10**5*np.eye(K)

        Ntot[0,:,reg] = np.sum(X0[:,:,reg],axis=1)

    return paramsTrue, scaleTrue, X0, Ntot

def extractParams(params,deadCells = []):
    ''' 
    Input:
        - params: K x (K+1) matrix of model parameters 
        - deadCells: optional, number of dead cells at each time point
    
    Output:
        - b: K-vector of birth rates for each cell type
        - A: K x K generator matrix 
    '''
    K = params.shape[0]
    b = params[:,0] # birth rates
    lam = params[:,1] # net growth rates
    nu = params[:,2:K+1] # switching rates

    nuAugm = np.zeros((K,K)) # augmented matrix based on (K-1) x (K-1) matrix nu
    for i in range(K):
        nuAugm[i,list(range(i))+list(range(i+1,K))] = nu[i]
    nuAugm = nuAugm - np.diag(np.sum(nuAugm,axis=1))
    A = np.diag(lam) + nuAugm # infinitesimal generator matrix

    if len(deadCells) > 0:
        A = np.vstack((np.hstack((A,b-lam)),np.zeros(K+1))) 

    return b,A