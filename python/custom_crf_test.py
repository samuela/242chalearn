from custom_crf_base import *
import numpy as np
import itertools as it
from random import choice

def bruteLogProb(ys, xl, theta, gamma):
    logprob = 0
    for t in range(T - 1):
        logprob += theta[ys[t], ys[t+1]]
    for t in range(T):
        for k in range(K):
            logprob += gamma[ys[t], k] * xl[k, t]
    return logprob

def brute_force_marg_logpart(xl, theta, gamma):
    joint_states = [(ys, bruteLogProb(ys, xl, theta, gamma))
                    for ys in it.product(range(D), repeat=T)]
    logPartition = np.log(np.sum([np.exp(lp) for ys, lp in joint_states]))
    
    bfp_1 = np.zeros([D, T])
    for t in range(T):
        for d in range(D):
            probs = [np.exp(logProb - logPartition) for (ys, logProb) in joint_states if ys[t] == d]
            bfp_1[d,t] = sum(probs)
            
    bfp_2 = np.zeros([D, D, T-1])
    for t in range(T-1):
        for i in range(D):
            for j in range(D):
                probs = [np.exp(logProb - logPartition) for (ys, logProb) in joint_states if (ys[t] == i and ys[t+1] == j)]
                bfp_2[i,j,t] = sum(probs)

    return bfp_1, bfp_2, logPartition

def brute_logLike(x,y,theta,gamma):
    ll = 0
    for xs, ys in zip(x, y):
        joint_states = [(ys_, bruteLogProb(ys_, xs, theta, gamma))
                        for ys_ in it.product(range(D), repeat=len(ys))]
        logPart = np.log(np.sum([np.exp(lp) for ys_, lp in joint_states]))
        ll += bruteLogProb(ys, xs, theta, gamma) - logPart
    return ll

def brute_posteriorMAP(xl, theta, gamma):
    joint_states = [(ys, bruteLogProb(ys, xl, theta, gamma))
                    for ys in it.product(range(D), repeat = T)]
#    ind = np.argmax([lp for (ys, lp) in joint_states])
#    return joint_states[ind][0]
    return max(joint_states, key = lambda x: x[1])[0]
    
# Log partition and marginals test cases:
K = 3
D = 3
T = 3
xl = np.random.randn(K*T).reshape(K,T)
theta = np.random.randn(D,D)
gamma = np.random.randn(D, K)
dpp_1, dpp_2, dplogPartition = calcMargsAndLogZ(xl, theta, gamma, D)
bfp_1, bfp_2, brutelogPartition = brute_force_marg_logpart(xl, theta,gamma)
print "marginals ", (np.abs(dpp_2 - bfp_2)).sum() + (np.abs(dpp_1 - bfp_1)).sum() + np.abs(dplogPartition - brutelogPartition)

# Log likelihood test cases:
K = 3
D = 3
T = 3
L = 5
x = [np.random.randn(K * T).reshape(K, T) for l in range(L)]
y = [np.array([choice(range(D)) for t in range(T)]) for l in range(L)]
theta = np.random.randn(D,D)
gamma = np.random.randn(D, K)
brute_ll = brute_logLike(x,y,theta,gamma)
dp_ll = logLikelihood(x,y,theta,gamma,D)
print "log likelihood ", np.abs(dp_ll - brute_ll)

import sys
sys.exit(0)

# Learn some parameters
K = 10
D = 10
T = 10
L = 10
x = [np.random.randn(K * T).reshape(K, T) for l in range(L)]
y = [np.array([choice(range(D)) for t in range(T)]) for l in range(L)]
theta = np.random.randn(D, D)
gamma = np.random.randn(D, K)
learned_theta, learned_gamma = learnParameters(x, y, K, D)
print learned_theta, "\n"
print learned_gamma

# Test posteriorMAP
for _ in range(10):
    K = 5
    D = 5
    T = 5
    xl = np.random.randn(K * T).reshape(K, T)
    theta = np.random.randn(D, D)
    gamma = np.random.randn(D, K)
    brute_map = brute_posteriorMAP(xl, theta, gamma)
    dp_map = posteriorMAP(xl, theta, gamma, D)
    print dp_map
    print brute_map

print samplePosterior(xl, theta, gamma, D, 10)
