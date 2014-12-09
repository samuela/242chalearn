from CRF_learning import *
import numpy as np
import itertools as it
from random import choice

K = 1
D = 2
T = 3
L = 1
# for _ in xrange(1000):

#xl = np.random.randn(K*T).reshape(K,T)
#theta = np.random.randn(D,D)
#gamma = np.random.randn(D, K)
#x = [np.random.randn(K*T).reshape(K,T) for l in range(L)]
x = [np.ones(K*T).reshape(K,T) for l in range(L)]
y = [np.array([choice(range(D)) for t in range(T)]) for l in range(L)]
#y = [np.array([1 for t in range(T)]) for l in range(L)]
#xl = np.arange(K*T).reshape([K,T])
#xl = np.array([0,1,0]).reshape([K,T])
#print xl
theta = np.arange(D*D).reshape([D,D])
#print theta
gamma = np.arange(D*K).reshape([D,K])
#print gamma



def bruteLogProb(ys, xl, theta, gamma):
    logprob = 0
    for t in range(T-1):
        logprob = logprob + theta[ys[t], ys[t+1]]
    for t in range(T):
        for k in range(K):
            logprob = logprob + gamma[ys[t], k] * xl[k, t]
    return logprob

def brute_force_marg_logpart(xl, theta, gamma):
    joint_states = [(ys, bruteLogProb(ys, xl, theta, gamma))
             for ys in it.product(range(D), repeat = T)]
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

    return bfp_1, bfp_2, ll, logPartition

#dpp_1, dpp_2, dplogPartition = marginals_and_logPartition(xl, theta, gamma, D)
#bfp_1, bfp_2, ll, brutelogPartition = brute_force_marg_logpart(xl, theta,gamma)

#print dpp_2, "\n"
#print bfp_2, "\n"
#print (np.abs(dpp_2 - bfp_2)).sum() + (np.abs(dpp_1 - bfp_1)).sum() + np.abs(dplogPartition - brutelogPartition)

def brute_logLike(x,y,theta,gamma):
    ll = 0
    for xs, ys in zip(x, y):
        joint_states = [(ys_, bruteLogProb(ys_, xs, theta, gamma))
                        for ys_ in it.product(range(D), repeat=len(ys))]
        logPart = np.log(np.sum([np.exp(lp) for ys_, lp in joint_states]))
        ll += bruteLogProb(ys, xs, theta, gamma) - logPart
    return ll

brute_ll = brute_logLike(x,y,theta,gamma)
dp_ll = logLikelihood(x,y,theta,gamma,D)

print dp_ll, "\n"
print brute_ll
