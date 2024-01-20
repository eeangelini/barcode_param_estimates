import numpy as np
import scipy.optimize as opt
from scipy.stats.distributions import chi2

from joblib import Parallel, delayed # for parallel loops

from bgCodeMLE import scaleConstraints, negLL


def ciLower(data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,idx,quant,tol):
    lbCheck = 0

    Aineq,bineq,Aeq,beq,lb,ub = scaleConstraints(Aineq,bineq,Aeq,beq,lb,ub,scaleMLE)
    thetaMLE = np.divide(thetaMLE,scaleMLE)

    K = thetaMLE.shape[0]

    thetaMLE = thetaMLE.flatten()
    scaleMLE = scaleMLE.flatten()
    lb = lb.flatten()
    ub = ub.flatten()

    myconstr = []
    if len(Aeq) > 0:
        Aeq.reshape((len(beq),K*(K+1)))
        lin_eq = opt.LinearConstraint(Aeq,beq,beq)
        myconstr.append(lin_eq)

    if len(Aineq) > 0:
        Aineq.reshape((len(bineq),K*(K+1)))
        lin_ineq = opt.LinearConstraint(Aineq,ub=bineq)
        myconstr.append(lin_ineq)
    
    nonlin_ineq = opt.NonlinearConstraint(lambda x: optLL + quant - negLL(x,scaleMLE,C,X0,data,T,dead),
                                          lb=np.zeros(quant.shape),
                                          ub=np.inf*np.ones(quant.shape))
    myconstr.append(nonlin_ineq)

    # print("Starting max likelihood step (CI lower bounds)...")
    res = opt.minimize(lambda x: x[idx], thetaMLE,
               constraints=myconstr, 
               options={'ftol': 1e-10, #'disp': True, 
                        'maxiter': 10000},
               bounds=opt.Bounds(lb,ub))
    
    # print("End max likelihood step (CI lower bounds)")
    sol = res.x

    lbYVals = negLL(sol,scaleMLE,C,X0,data,T,dead)
    sol = sol*scaleMLE
    lbCI = sol[idx]

    if (abs(lbYVals - optLL - quant)/(optLL+quant)<tol) or (lbCI == lb[idx] and lbYVals < optLL + quant + tol):
        lbCheck = 1

    return lbCI,lbYVals,lbCheck

def ciUpper(data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,idx,quant,tol):
    ubCheck = 0

    Aineq,bineq,Aeq,beq,lb,ub = scaleConstraints(Aineq,bineq,Aeq,beq,lb,ub,scaleMLE)
    thetaMLE = np.divide(thetaMLE,scaleMLE)

    K = thetaMLE.shape[0]

    thetaMLE = thetaMLE.flatten()
    scaleMLE = scaleMLE.flatten()
    lb = lb.flatten()
    ub = ub.flatten()

    myconstr = []
    if len(Aeq) > 0:
        Aeq.reshape((K*(K+1),len(beq)))
        lin_eq = opt.LinearConstraint(Aeq,beq,beq)
        myconstr.append(lin_eq)

    if len(Aineq) > 0:
        Aineq.reshape((K*(K+1),len(bineq)))
        lin_ineq = opt.LinearConstraint(Aineq,ub = bineq)
        myconstr.append(lin_ineq)
    
    nonlin_ineq = opt.NonlinearConstraint(lambda x: optLL + quant - negLL(x,scaleMLE,C,X0,data,T,dead),
                                          lb=np.zeros(quant.shape),
                                          ub=np.inf*np.ones(quant.shape))
    myconstr.append(nonlin_ineq)

    # print("Starting max likelihood step (CI upper bounds)...")
    res = opt.minimize(lambda x: -x[idx], thetaMLE,
               constraints=myconstr, 
               options={'ftol': 1e-10, #'disp': True, 
                        'maxiter': 10000},
               bounds=opt.Bounds(lb,ub))
    
    # print("End max likelihood step (CI upper bounds)")
    sol = res.x

    ubYVals = negLL(sol,scaleMLE,C,X0,data,T,dead)
    sol = sol*scaleMLE
    ubCI = sol[idx]

    if (abs(ubYVals - optLL - quant)/(optLL+quant)<tol) or (ubCI == ub[idx] and ubYVals < optLL + quant + tol):
        ubCheck = 1

    return ubCI,ubYVals,ubCheck

def loopCI(idx,ciOpt,nOpt,data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,quant,tol):
    ''' This is the bulk of code that calls ciLower and ciUpper to get the confidence intervals,
     and checks the appropriate exit conditions. Function above (getCI) calls this function in parallel
     over index idx (i.e., compute CIs for each parameter in parallel).'''
    if ciOpt[idx] == 1:
        lbCI = np.inf
        ubCI = -np.inf
        
        for n in range(nOpt+1):
            end_cond = 0
            # Get lower bound   
            while end_cond == 0:
                ci_num_left_temp, ci_yvalues_num_left_temp, ci_check_num_left_temp = ciLower(data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,idx,quant,tol)

                # relative error b/t negative log likelihood at returned endpoint 
                # and (-LL at MLE parameter estimate + quant) is within tol AND
                # returned endpoint is to left of MLE parameter estimate
                if ci_check_num_left_temp == 1 and ci_num_left_temp <= thetaMLE.flatten()[idx]:
                    end_cond += 1

            # If computing multiple times, update if above process returned a lower lower bound.
            # If just computing once, condition is always True (set ci_num[0,i,j] = np.inf),
            # updates value of ci_num[0,i,j] accordingly.
            if ci_num_left_temp <= lbCI:
                lbCI = ci_num_left_temp
                yValsLB = ci_yvalues_num_left_temp
                checkLB = ci_check_num_left_temp

            # Now, get upper bound
            end_cond = 0   
            while end_cond == 0:
                ci_num_right_temp, ci_yvalues_num_right_temp, ci_check_num_right_temp = ciUpper(data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,idx,quant,tol)

                # relative error b/t negative log likelihood at returned endpoint 
                # and (-LL at MLE parameter estimate + quant) is within tol AND
                # returned endpoint is to right of MLE parameter estimate
                if ci_check_num_right_temp and ci_num_right_temp >= thetaMLE.flatten()[idx]:
                    end_cond += 1

            # If computing multiple times, update if above process returned a higher upper bound.
            # If just computing once, condition is always True (set ci_num[1,i,j] = -np.inf),
            # updates value of ci_num[1,i,j] accordingly.
            if ci_num_right_temp >= ubCI:
                ubCI = ci_num_right_temp
                yValsUB = ci_yvalues_num_right_temp
                checkUB = ci_check_num_right_temp

    # return relevant values as a list
    return [lbCI, yValsLB, checkLB, ubCI, yValsUB, checkUB]

def getCI(alpha,ciOpt,nOpt,data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,tol):
    K = thetaMLE.shape[0]
    quant = chi2.ppf(1-alpha,df=1)

    CI = np.zeros((2,K*(K+1)))
    yValsCI = np.zeros((2,K*(K+1)))
    checkCI = np.zeros((2,K*(K+1)))

    # Compute confidence intervals for each parameter in parallel
    tempFunc = lambda idx: loopCI(idx,ciOpt.flatten(),nOpt,data,dead,C,X0,T,thetaMLE,scaleMLE,optLL,Aineq,bineq,Aeq,beq,lb,ub,quant,tol)
    resultList = Parallel(n_jobs=-1)(delayed(tempFunc)(i) for i in range(K*(K+1)))

    for i in range(K*(K+1)):
        CI[0,i] = resultList[i][0]
        CI[1,i] = resultList[i][3]
        yValsCI[0,i] = resultList[i][1]
        yValsCI[1,i] = resultList[i][4]
        checkCI[0,i] = resultList[i][2]
        checkCI[1,i] = resultList[i][5]
                        
    return np.reshape(CI,(2,K,K+1)), yValsCI, checkCI