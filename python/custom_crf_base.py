# We will calculate the log likelihood and gradient of the log likelihood for
# any given parameters theta and gamma and training exampes (x,y)

# "Global" variables
# L is number of training examples
# K is number of features
# D is number of labels

import numpy as np
import scipy.optimize as opt
from random import sample
#import multiprocessing as mp
#from multiprocessing import Pool
#from multiprocessing.dummy import Pool as ThreadPool

def suffStats(x, y, K, D):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels

    L = len(x)
    gamma_ss = np.zeros([D,K])
    theta_ss = np.zeros([D,D])
    for l in range(L):
        # first suff stat for gamma
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        gamma_ss += np.dot(ymat, x[l].T)

        # next suff stats for theta
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        theta_ss += np.dot(yt , ytp1.T)

    return gamma_ss, theta_ss

def expectedStats(x, theta, gamma, K, D):
    # x is an L length list of K by T(l) matrices of features
    # theta is a D by D array
    # gamma is a D by K array

    L = len(x)
    gamma_es = np.zeros([D,K])
    theta_es = np.zeros([D,D])
    for l in range(L):
        p_1, p_2, _ = marginals_and_logPartition(x[l], theta, gamma, D) #p_1 is a D by T(l) array of single y marignals, p_2 is a D by D by T(l)-1 array of y pair marginals
        # first stats for gamma
        gamma_es = gamma_es + np.dot(p_1, x[l].T)
        # next for theta
        theta_es = theta_es + np.sum(p_2, axis=2)
    return (gamma_es, theta_es)

def gradient(x, y, theta, gamma, K, D):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    # theta is a D by D array
    # gamma is a D by K array

    gamma_ss, theta_ss = suffStats(x,y, K, D)
    gamma_es, theta_es = expectedStats(x,theta,gamma, K, D)
    return paramsToVector(theta_ss - theta_es, gamma_ss - gamma_es)

def logLikelihood(x, y, theta, gamma, D):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    # theta is a D by D array
    # gamma is a D by K array

    L = len(x)
    def calculate_term(l):
        ans = 0
        # theta term
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        print y, ymat
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        n_ij = np.dot(yt, ytp1.T)
        theta_term = (theta * n_ij).sum()

        # gamma term
        gamma_term = (np.dot(ymat, x[l].T) * gamma).sum()

        # log partition function
        _, _, logPartition = marginals_and_logPartition(x[l],theta,gamma, D)
        return theta_term + gamma_term - logPartition

    return sum(map(calculate_term, range(L)))


def logPhi(t, xl, theta, gamma, D):
    # xl is a K by T matrix of features
    # t is an integer between 1 and T
    # theta is a D by D array
    # gamma is a D by K array
    # returns a matrix A such that A[i,j] = phi(y_t = i, y_{t+1} = j)
    if (t == 1):
        return theta + np.dot(gamma, xl[:, 0]).reshape(D,1) + np.dot(gamma, xl[:,1]).reshape(1,D)
    else:
        return theta + np.dot(gamma, xl[:, t]).reshape(1,D)

def marginals_and_logPartition(xl, theta, gamma, D):
    # xl is a K by T matrix of features
    # theta is a D by D array
    # gamma is a D by K array

    _, T = xl.shape
    m_fwd = np.ones([D, T]) #forward messages. First column is just ones
    # m_fwd[:,t] = m_{t,t+1}
    log_sums = np.zeros(T) #logs of the normalization constants
    for t in range(1,T):
        lp = logPhi(t, xl, theta, gamma, D)
        max_val = np.max(lp)
        unnormalized_message = np.dot(np.exp(lp-max_val).T, m_fwd[:, t-1])
        sum_t = sum(unnormalized_message)
        m_fwd[:,t] = unnormalized_message / sum_t
        log_sums[t] = np.log(sum_t) + max_val

    m_bwd = np.ones([D, T]) #backward messages. Last column is just ones
    # m_bwd[:,t] = m_{t+2,t+1}
    for t in range(T - 2, -1, -1): #T-2,T-3,...,0
        lp = logPhi(t+1, xl, theta, gamma, D)
        max_val = np.max(lp)
        unnormalized_message = np.dot(np.exp(lp-max_val), m_bwd[:, t+1])
        m_bwd[:,t] = unnormalized_message / sum(unnormalized_message)

    logPartition = np.sum(log_sums) #log partition function
    p_1 = m_fwd * m_bwd #single node marginals unnormalized
    p_1 = p_1 / p_1.sum(axis=0) #normalized

    #now pairwise marginals
    p_2 = np.zeros([D, D, T-1])
    for t in range(T-1):
        lp = logPhi(t+1, xl, theta, gamma, D)
        max_val = np.max(lp)
        p_2[:,:,t] = np.exp(lp-max_val) * m_fwd[:,t].reshape(D,1) * m_bwd[:,t+1].reshape(1,D)# this is the unnormalized marginal on y_{t+1,t+2}
        p_2[:,:,t] = p_2[:,:,t] / p_2[:,:,t].sum() #normalized

    return p_1, p_2, logPartition

def paramsToVector(theta, gamma):
    return np.concatenate((theta.ravel(), gamma.ravel()))

def vectorToParams(vec, K, D):
    theta = vec[:D*D].reshape(D,D)
    gamma = vec[D*D:].reshape(D,K)
    return theta, gamma

def learnParameters(x, y, K, D, regularization=1, theta_init=None, gamma_init=None):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    # theta_init is a D by D array
    # gamma_init is a D by K array

    if theta_init is None:
        theta_init = np.random.randn(D, D)
    if gamma_init is None:
        gamma_init = np.random.randn(D, K)

    def nll(vec):
        theta, gamma = vectorToParams(vec, K, D)
        return (-1.0) * (logLikelihood(x, y, theta, gamma, D) - regularization * np.dot(vec,vec))

    def ngrad(vec):
        theta, gamma = vectorToParams(vec, K, D)
        return (-1.0) * (gradient(x, y, theta, gamma, K, D) - 2 * regularization * vec)

    results = opt.fmin_l_bfgs_b(nll, paramsToVector(theta_init, gamma_init), ngrad,
                                iprint=0, epsilon=1e-4)
    return vectorToParams(results[0], K, D)

def learnSGD(x, y, theta_init, gamma_init, lamb, K, D):
    max_iter = 500
    tol = 1e-4
    batch_size = 50

    for i in max_iter:
        batch_x, batch_y = zip(*sample(zip(x, y), batch_size))
        # TODO

def posteriorMAP(xl, theta, gamma, D):
    #TODO
    return 0

def samplePosterior(xl, theta, gamma, D, num_samples):
    #TODO
    return 0

def posteriorMarginalMAP(xl, theta, gamma, D):
    p_1, _, _ = marginals_and_logPartition(xl, theta, gamma, D) #p_1 is a D by T matrix
    return np.argmax(p_1, axis=0)
