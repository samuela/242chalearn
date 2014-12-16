import numpy as np
import scipy.optimize as opt
from random import sample, choice

import multiprocessing as mp
import time


# "Global" variables
# L is number of training examples
# K is number of features
# D is number of labels

def suffStats(x, y, K, D):
    """Calculate the sufficient statistics of the data x, y.

    Args:
        x : an L length list of K x T(l) matrices of features
        y : an L length list of T(l) length arrays of labels

    Returns:
        theta_ss : a D x D matrix of sufficient statistics for the theta
            parameters.
        gamma_ss : a D x K matrix of sufficient statistics for the gamma
            parameters.
    """
    L = len(x)
    gamma_ss = np.zeros([D,K])
    theta_ss = np.zeros([D,D])
    for l in xrange(L):
        # first suff stat for gamma
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        gamma_ss += np.dot(ymat, x[l].T)

        # next suff stats for theta
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        theta_ss += np.dot(yt , ytp1.T)

    return theta_ss, gamma_ss

def expectedStats(x, theta, gamma, K, D, calcMarginals=None):
    """Calculate the expected statistics of the data x under the parameters
    theta and gamma.

    Args:
        x : an L length list of K x T(l) matrices of features.
        theta : D x D matrix of the theta parameters.
        gamma : D x K matrix of the gamma parameters.

    Returns:
        theta_es : a D x D matrix of expected statistics for the theta
            parameters under the current parameters.
        gamma_ss : a D x K matrix of expected statistics for the gamma
            parameters under the current parameters.
    """
    L = len(x)
    gamma_es = np.zeros([D,K])
    theta_es = np.zeros([D,D])
    for l in xrange(L):
        # p_1 is a D by T(l) array of single y marignals
        # p_2 is a D by D by T(l)-1 array of y pair marginals
        if calcMarginals:
            p_1, p_2, _ = calcMarginals(l, theta, gamma)
        else:
            p_1, p_2, _ = calcMargsAndLogZ(x[l], theta, gamma, D)

        # first stats for gamma
        gamma_es += np.dot(p_1, x[l].T)
        # next for theta
        theta_es += np.sum(p_2, axis=2)

    return theta_es, gamma_es

def gradient(x, y, theta, gamma, K, D, calcMarginals=None):
    """Computes the gradient of the log-likelihood.

    Args:
        x : an L length list of K x T(l) matrices of features.
        y : an L length list of T(l) length arrays of labels.
        theta : D x D matrix of the theta parameters.
        gamma : D x K matrix of the gamma parameters.
        K : the number of features.
        D : the number of latent labels.
        calcMarginals : function to calculate the single/pair-node marginal
            distributions and the log partition. Defaults to calcMargsAndLogZ.

    Returns:
        A combined gradient vector of size D * D + D * K.
    """
    theta_ss, gamma_ss = suffStats(x,y, K, D)
    theta_es, gamma_es = expectedStats(x, theta, gamma, K, D, calcMarginals)
    return paramsToVector(theta_ss - theta_es, gamma_ss - gamma_es)

def logLikelihood(x, y, theta, gamma, D, calcMarginals=None):
    """Calculate the log-likelihood of the data x, y under the model parameters
    theta and gamma.

    Args:
        x : an L length list of K x T(l) matrices of features.
        y : an L length list of T(l) length arrays of labels.
        theta : D x D matrix of the theta parameters.
        gamma : D x K matrix of the gamma parameters.
        D : the number of latent labels.
        calcMarginals : function to calculate the single/pair-node marginal
            distributions and the log partition. Defaults to calcMargsAndLogZ.

    Returns:
        The log-likelihood.
    """
    L = len(x)
    ll = 0
    for l in xrange(L):
        # theta term
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        n_ij = np.dot(yt, ytp1.T)
        theta_term = (theta * n_ij).sum()

        # gamma term
        gamma_term = (np.dot(ymat, x[l].T) * gamma).sum()

        # log partition function
        if calcMarginals:
            _, _, logPartition = calcMarginals(l, theta, gamma)
        else:
            _, _, logPartition = calcMargsAndLogZ(x[l], theta, gamma, D)

        ll += theta_term + gamma_term - logPartition

    return ll

def logPhi(t, xl, theta, gamma, D):
    """Calculate a clique function potential under the given model parameters.

    Args:
        xl : a K by T matrix of features.
        t : an integer between 1 and T.
        theta : D x D matrix of the theta parameters.
        gamma : D x K matrix of the gamma parameters.
        D : the number of latent labels.

    Returns:
        Matrix A such that A[i,j] = phi(y_t = i, y_{t+1} = j).
    """
    if t == 1:
        return theta + np.dot(gamma, xl[:, 0]).reshape(D, 1) \
                     + np.dot(gamma, xl[:, 1]).reshape(1, D)
    else:
        return theta + np.dot(gamma, xl[:, t]).reshape(1, D)

# @profile
def calcMargsAndLogZ(xl, theta, gamma, D):
    """Calculate the single/pair-node marginals and the log partition under the
    given model parameters. Requires that len(xl) > 1.

    Args:
        xl : a K by T matrix of features.
        theta : D x D matrix of the theta parameters.
        gamma : D x K matrix of the gamma parameters.

    Returns:
        p_1 : D x T array of signle-node marginals.
        p_2 : D x D x (T-1) array of node-pair marginals.
        logPartition : log(Z)
    """
    _, T = xl.shape

    logPhi = theta.reshape(D, D, 1) + np.dot(gamma, xl).reshape(1, D, T)
    logPhi[:,:,1] += np.dot(gamma, xl[:, 0]).reshape(D, 1)

    logPhiMaxes = np.max(np.max(logPhi, axis=0), axis=0).reshape(1, 1, T)
    logPhiExp = np.exp(logPhi - logPhiMaxes)

    m_fwd = np.ones([D, T]) # forward messages. First column is just ones
    # m_fwd[:,t] = m_{t,t+1}
    logPartition = 0.0 # log partition function
    for t in xrange(1, T):
#        lp = logPhi(t, xl, theta, gamma, D) # exp sum log trick
#         lp = logPhi[:,:,t]
#        max_val = np.max(lp)
#        max_val = logPhiMaxes[t]

        unnorm_message = np.dot(logPhiExp[:,:,t].T, m_fwd[:, t-1])
#        unnorm_message = np.dot(np.exp(lp - max_val).T, m_fwd[:, t-1])
        sum_t = np.sum(unnorm_message)
        m_fwd[:,t] = unnorm_message / sum_t
        logPartition += np.log(sum_t) + logPhiMaxes[:,:,t]
#        logPartition += np.log(sum_t) + max_val

    m_bwd = np.ones([D, T]) # backward messages. Last column is just ones
    # m_bwd[:,t] = m_{t+2,t+1}
    for t in xrange(T - 2, -1, -1): #T-2,T-3,...,0
#        lp = logPhi(t+1, xl, theta, gamma, D) # exp sum log trick
#        lp = logPhi[:,:,t+1]
#        max_val = np.max(lp)
#        max_val = logPhiMaxes[t + 1]
        unnorm_message = np.dot(logPhiExp[:,:,t+1], m_bwd[:, t+1])
#        unnorm_message = np.dot(np.exp(lp - max_val), m_bwd[:, t+1])
        m_bwd[:,t] = unnorm_message / np.sum(unnorm_message)

    p_1 = m_fwd * m_bwd # single node marginals unnormalized
    p_1 /= p_1.sum(axis=0) # normalized
#    p_1 = p_1 / p_1.sum(axis=0) # normalized

    #now pairwise marginals
    p_2 = np.zeros([D, D, T - 1])
    for t in xrange(T - 1):
#        lp = logPhi(t + 1, xl, theta, gamma, D) # exp sum log trick
#        lp = logPhi[:,:,t+1]
#        max_val = np.max(lp)
#        max_val = logPhiMaxes[t + 1]
        # this is the unnormalized marginal on y_{t+1,t+2}
        p_2[:,:,t] = logPhiExp[:,:,t+1] * m_fwd[:,t].reshape(D, 1) \
                                        * m_bwd[:,t+1].reshape(1, D)
        p_2[:,:,t] = p_2[:,:,t] / np.sum(p_2[:,:,t]) # normalized

    return p_1, p_2, logPartition

def _calcMargs(args):
    """Wrapper for calcMargsAndLogZ required by pool.map()."""
    return calcMargsAndLogZ(*args)

def paramsToVector(theta, gamma):
    """Converts the theta, gamma parameter matrices into a vector for the
    optimization algorithm."""
    return np.concatenate((theta.ravel(), gamma.ravel()))

def vectorToParams(vec, K, D):
    """Recovers the theta, gamma parameter matrices from vector form."""
    theta = vec[:D*D].reshape(D, D)
    gamma = vec[D*D:].reshape(D, K)
    return theta, gamma

def learnParameters(x, y, K, D, lamb=1.0, theta_init=None, gamma_init=None,
                    maxiter=100):
    """Learn the model parameters with the L-BFGS algorithm.

    Args:
        x : an L length list of K x T(l) matrices of features.
        y : an L length list of T(l) length arrays of labels.
        K : the number of features.
        D : the number of latent labels.
        lamb : the L2 loss penalty.
        theta_init : the initial guess at the theta parameters. Defaults to
            zeros.
        gamma_init : the initial guess at the gamma parameters. Defaults to
            zeros.
        maxiter : the maximum number of iterations to run.

    Returns:
        Who knows right now.
    """
    L = len(x)

    if theta_init is None:
        theta_init = np.zeros((D, D))
    if gamma_init is None:
        gamma_init = np.zeros((D, K))

    # memos is a dict mapping (theta, gamma) -> [(p_1, p_2, logZ), ...]
    # When calcMarginals is called, it first checks whether the calculation has
    # already been performed for the given theta, gamma parameters. If so, it
    # returns the cached values. Otherwise, we calculate the marginals and log
    # partition values over all sequences in parallel.
    memos = {}
    def calcMarginals(l, theta, gamma):
        key = (hash(theta.tostring()), hash(gamma.tostring()))
        if key in memos:
            # print 'Cached!'
            return memos[key][l]
        else:
            # print 'Missing'

            # Clear the memos dict, so we don't blow up memory.
            memos.clear()

            # Calculate all in parallel
            memos[key] = pool.map(_calcMargs,
                                  zip(x, [theta] * L, [gamma] * L, [D] * L))

            return memos[key][l]

    def nll(vec):
        theta, gamma = vectorToParams(vec, K, D)
        return (-1.0) * (logLikelihood(x, y, theta, gamma, D, calcMarginals)
                         - lamb * np.dot(vec, vec))

    def ngrad(vec):
        theta, gamma = vectorToParams(vec, K, D)
        return (-1.0) * (gradient(x, y, theta, gamma, K, D, calcMarginals)
                         - 2 * lamb * vec)

    ll_per_iter = []
    last_time = [time.clock()]
    def callback(vec):
        theta, gamma = vectorToParams(vec, K, D)
        ll = logLikelihood(x, y, theta, gamma, D, calcMarginals)
        ll_per_iter.append(ll)

        now = time.clock()
        print 'Iter', len(ll_per_iter), '\t',
        print 'LL:', ll, '\t',
        print 'L2 Loss:', lamb * np.sqrt(np.sum(np.power(vec, 2))), '\t',
        print 'Time:', now - last_time[0]

        # Clear the memos dict, so we don't blow up memory.
        memos.clear()

        last_time[0] = now

        import resource
        print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'Mb'


    opt_results = opt.fmin_l_bfgs_b(nll, paramsToVector(theta_init, gamma_init), ngrad,
                                    iprint=0, epsilon=1e-3, callback=callback,
                                    maxiter=maxiter)
    return opt_results, ll_per_iter

def learnSGD(x, y, K, D, lamb=1.0, theta_init=None, gamma_init=None, maxiter=500):
    L = len(x)
    tol = 1e-2
    batch_size = 10

    last_nll = None
#     step_size = 5 * 1e-7

    if theta_init is None:
        theta_init = np.zeros((D, D))
    if gamma_init is None:
        gamma_init = np.zeros((D, K))

    theta, gamma = theta_init, gamma_init

    for i in xrange(maxiter):
        batch_x, batch_y = zip(*sample(zip(x, y), batch_size))

        opt_results, _ = learnParameters(x, y, K, D, lamb=lamb, theta_init=theta, gamma_init=gamma, maxiter=1)

        new_nll = nll(vec)
        print 'Iter', i, new_nll, alpha * np.sqrt(np.sum(np.power(grad, 2)))

        if last_nll != None and np.abs(last_nll - new_nll) < tol:
            break

        last_nll = new_nll

def posteriorMAP(xl, theta, gamma, D):
    # xl is a K by T matrix of features
    # theta is a D by D array
    # gamma is a D by K array

    _, T = xl.shape
    R = np.zeros([T, D])
    M = np.zeros([T, D])
    for t in range(1,T):
        lp = logPhi(t, xl, theta, gamma, D)
        objective = lp + M[t-1, :].reshape(D,1)
        R[t, :] = np.argmax(objective, axis=0)
        M[t, :] = np.max(objective, axis=0)
    y = np.zeros(T)
    y[T-1] = np.argmax(M[T-1, :])
    for t in range(T-2,-1,-1):
       y[t] = R[t+1, int(y[t+1])]
    return y

def samplePosterior(xl, theta, gamma, D, num_samples):
    # xl is a K by T matrix of features
    # theta is a D by D array
    # gamma is a D by K array
    # num_samples is the number of samples to draw from the posterior distribution

    _, T = xl.shape
    S = np.ones([T, D])
    for t in range(T-2, -1, -1):
        lp = logPhi(t-1, xl, theta, gamma, D) # exp sum log trick
        max_val = np.max(lp)
        S[t, :] = np.dot(np.exp(lp-max_val), S[t+1, :].T)
        S[t, :] = S[t,:] / S[t, :].sum()
    samples = np.zeros([num_samples, T])
    for num in range(num_samples):
        P = np.zeros([T, D])
        P[0, :] = S[0, :]
        samples[num, 0] = np.random.choice(range(D), p = P[0, :])
        for t in range(1,T):
            lp = logPhi(t, xl, theta, gamma, D) # exp sum log trick
            max_val = np.max(lp)
            P[t,:] = np.exp(lp - max_val)[samples[num, t-1], :] * S[t, :]
            P[t,:] = P[t,:] / P[t,:].sum()
            samples[num, t] = np.random.choice(range(D), p = P[t, :])
    return samples

def posteriorMarginalMAP(xl, theta, gamma, D):
    p_1, _, _ = calcMargsAndLogZ(xl, theta, gamma, D) #p_1 is a D by T matrix
    return np.argmax(p_1, axis=0)

pool = mp.Pool()
