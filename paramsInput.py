import numpy as np

# K1 is the number of phenotypes = number barcode lineages in the population
K = 2
# R is the number of experimental replicates performed.
R = 3
# T is the row vector of experimental timepoints.
T = np.arange(1,7)
# L is the number of experimental timepoints.
L = len(T)

# True parameters for artificially simulated data: 
# # [b_1, lam_1, nu_12, nu_13, ... nu_1K; 
# #  b_2, lam_2, nu_21, nu_23, ... nu_2K; 
#               ...   
#    b_K, lam_K, nu_K1, nu_K2, ... nu_K,K-1]
paramsTrue = np.array([[0.6,0.3,0.02],[1.0,0.5,0.04]]) 

lam = paramsTrue[:,1] # [lambda_1, lambda_2, ... , lambda_k]
nu = paramsTrue[:,2:K+1]

# initialize true rate matrix:
# a_ij = nu_ji
ATrue = np.zeros((K,K))
for i in range(K):
    ATrue[list(range(i))+list(range(i+1,K)),i] = nu[i] 

# diagonal elements : lambda_i - sum_{j!=i} a_ji
ATrue = ATrue - np.diag(ATrue.sum(axis=0))
ATrue = np.diag(lam) + ATrue

# for i in range(K):
#     mySum = 0
#     for j in range(K):
#         if i != j
#             mySum += ATrue[j,i]
#     ATrue[i,i] = growthRates[i] - mySum

# N is an K x K matrix that contains the starting number of
# cells of each type x barcode (used for branching process)
X0 = np.array([[1e3,0],[0,1e3]])
