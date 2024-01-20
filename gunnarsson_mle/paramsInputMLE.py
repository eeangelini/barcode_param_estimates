import numpy as np

# K is the number of phenotypes in the model.
K = 2
# R is the number of experimental replicates performed.
R = 3
# T is the row vector of experimental timepoints.
T = np.arange(1,7)
# L is the number of experimental timepoints.
L = len(T)

# True parameters for artificially simulated data: 
# # [b_1,lam_1,nu_12; b_2, lam_2, nu_21]
paramsTrue = np.array([[0.6,0.3,0.02],[1.0,0.5,0.04]]) 

# The user has the option of supplying an I x 1 x L x R tensor with the
# number of dead cells at each time point. In that case, the estimation is
# performed on the agumented model shown in Figure 6 of the Gunnarsson et al.
# paper, where a new state representing dead cells is added.
# Note that the augmented model does not take into account clearance of
# dead cells. See Section "Improving identifiability of the rates of cell
# division and cell death" of the Gunnarsson et al. paper.
deadCells = []

# lowerBounds is a K x (K+1) matrix which holds lower bounds for the branching
# process model parameters. The arrangement of parameters is as described
# at the top of the script. For example, the first column holds lower
# bounds for the birth rates of the phenotypes.
lowerBounds = np.array([[0,-4,1e-20],[0,-4,1e-20]])
# **Note** was having issues with covariance matrix estimate being singular
# during MLE step when one or either of transition rates were allowed to equal 0
# here, forcing them to be "non-zero" (but could effectively be zero)

# upperBounds is a K x (K+1) matrix which holds upper bounds for the branching
# process model parameters.
upperBounds = np.array([[2,4,1],[2,4,1]])

# Aineq is an n x K x (K+1) array, where n is the number of linear
# inequality constraints on the branching process model parameters imposed
# by the user. Each inequality constraint is of the form
# a_1x_1 + ... + a_Mx_M <= b,
# where x_1,...,x_M are the branching process model parameters.
# For each m = 1,...,n, Aineq(:,:,m) holds the coefficients of the
# left-hand side of the m-th inequality constraint. The arrangement of
# parameters is as described at the top of the script.
Aineq = np.array([])

# bineq_num is an n x 1 vector, where n is the number of linear inequality
# constraints. For each m = 1,...,n, bineq_num(m) holds the coefficient of
# the right-hand side of the m-th inequality constraint.
bineq = np.array([])

# Aeq_num is an n x K x (K+1) array, where n is the number of linear
# equality constraints on the branching process model parameters imposed
# by the user. Each equality constraint is of the form
# a_1x_1 + ... + a_Mx_M = b,
# where x_1,...,x_M are the branching process model parameters.
# For each m = 1,...,n, Aeq_num(:,:,m) holds the coefficients of the
# left-hand side of the m-th equality constraint.
Aeq = np.array([])

# beq_num is an n x 1 vector, where n is the number of linear equality
# constraints. For each m = 1,...,n, beq_num(m) holds the coefficient of
# the right-hand side of the m-th equality constraint.
beq = np.array([])

# In this script, neither an inequality constraint nor an equality
# constraint is implemented.

# theta0_def is a K x (K+1) x n tensor, which can be used to supply
# initial guesses for the MLE optimization problem. Here, n is the number
# of distinct initial guesses the user wishes to supply. The arrangement of
# parameters is as described at the top of the script.
theta0def = []

# By default, the MLE optimization problem is solved once, using an initial
# guess based on simple deterministic parameter estimates. Here, the user
# is given the option to request that the optimization problem is solved
# for multiple initial guesses, each based on the simple deterministic
# estimates. See Appendix "Implementation in MATLAB" of the Gunnarsson
# paper.
nOptSimple = 0

# Here, the user is given the option to request that the MLE optimization
# problem for is solved for multiple initial guesses, where the initial
# guesses are chosen in a random fashion as described in Appendix
# "Generation of artificial data" of the Gunnarsson paper.
nOptRandom = 0

# ciOption is a K x (K+1) matrix holding zeros and ones. Setting
# ciOption(i,j) = 1 will compute a confidence interval for parameter (i,j)
# of the branching process model, where the arrangement of parameters is
# as described at the top of the script. 
ciOption = np.ones((K,K+1))

# alpha_q sets the level for the confidence intervals.
alpha_q = 0.05

# nOptCI allows the user to request that confidence intervals are computed
# multiple times, starting from different initial guesses.
nOptCI = 0

# If dataVis is set to True, plots are produced that show for each
# initial condition and each type how well the statistical model fits the
# data. More precisely, the mean prediction of the statistical model and
# 1-alpha_q confidence intervals, under the assumption that the MLE
# estimates are the true parameters, are compared with the data.
dataVis = True

## Show initial guess and scale: toggle on off
printInit = True

## Show coefficients for constraints and bounds when doing MLE step
printBounds = False

## Data is with cell barcodes or not (i.e., cell line experiments)
withBarcodes = True

## Treat barcode lineages as indistinguishable for MLE step (i.e., just use for initial guess)
poolBCs = False

# cell line experiments (Gunnarsson et al): each row gives initial conditions across cell types 
# barcode data: initial conditions for (barcode i) x (type j)
# X0 = np.array([[1e3,0],[0,1e3]]) # 'Barcodes 1 & 2' (poolBCs True & False), Gunnarsson et al
# X0 = np.array([[990,10],[10,990]]) # 'Barcodes 3'
X0 = np.array([[900,100],[100,900]]) # 'Barcodes 4'

# Number of initial conditions
if withBarcodes and poolBCs:
    I = 1
    X0bc = X0.copy() # full initial conditions by barcodes
    X0 = np.sum(X0,axis = 0)[None,:] # sum over barcodes to just get cell type initial conditions (want row vector)
else:
    # I is the number of initial conditions used in the cell line experiments
    # OR treat barcode lineages as two independent cell line experiments
    I = X0.shape[0]

# C is a list, where C[i] is a K x J_i matrix which
# allows the user to reduce the experimental data under the i-th initial
# condition. This option can be useful for reducible switching dynamics.
# See Appendices "Estimation for reducible switching dynamics" and
# "Implementation in MATLAB" of the Gunnarsson paper.
C = []
for i in range(I):
    C.append(np.eye(K))

## Tolerance confidence intervals and quantiles
tol = 1e-3