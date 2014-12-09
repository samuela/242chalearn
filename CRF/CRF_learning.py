# We will calculate the log likelihood and gradient of the log likelihood for
# any given parameters theta and gamma and training exampes (x,y)

# L is number of training examples
K = 140 # K is number of features
D = 21 # D is number of labels

import numpy as np

def suffStats(x,y):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    
    L = len(x)
    gamma_ss = np.zeros([D,K])
    theta_ss = np.zeros([D,D])
    for l in range(L):
        # first suff stat for gamma
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        gamma_ss = gamma_ss + np.dot(ymat, x[l].T)
    
        # next suff stats for theta
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        theta_ss = theta_ss + np.dot(yt , ytp1.T)
    return (gamma_ss, theta_ss)

def expectedStats(x, theta, gamma):
    # x is an L length list of K by T(l) matrices of features
    # theta is a D by D array
    # gamma is a D by K array

    L = len(x)
    gamma_es = np.zeros([D,K])
    theta_es = np.zeros([D,D])
    for l in range(L):
        p_1, p_2, _ = marginals_and_logPartition(x[l], theta, gamma) #p_1 is a D by T(l) array of single y marignals, p_2 is a D by D by T(l)-1 array of y pair marginals
        # first stats for gamma
        gamma_es = gamma_es + np.dot(p_1, x[l].T)
        # next for theta
        theta_es = theta_es + np.sum(p_2, axis=2)
    return (gamma_es, theta_es)

def gradient(x,y,theta,gamma):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    # theta is a D by D array
    # gamma is a D by K array

    gamma_ss, theta_ss = suffStats(x,y)
    gamma_es, theta_es = expectedStats(x,theta,gamma)
    return np.concatenate(((gamma_ss-gamma_es).ravel(), (theta_ss-theta_es).ravel()))

def logLikelihood(x,y,theta,gamma):
    # x is an L length list of K by T(l) matrices of features
    # y is an L length list of T(l) length arrays of labels
    # theta is a D by D array
    # gamma is a D by K array

    L = len(x)
    ans = 0
    for l in range(L):
        # theta term
        ymat = np.array([y[l] == i for i in range(D)]) * 1.0
        yt = ymat[:,:-1]
        ytp1 = ymat[:,1:]
        n_ij = np.dot(yt, ytp1.T)
        ans = ans + (theta * n_ij).sum()

        # gamma term
        ans = ans + (np.dot(ymat, x[l].T) * gamma).sum()

        # log partition function
        _, _, logPartition = marginals_and_logPartition(x[l],theta,gamma)
        ans = ans - logPartition
    return ans

def phi(t, xl, theta, gamma):
    # xl is a K by T matrix of features
    # t is an integer between 1 and T
    # theta is a D by D array
    # gamma is a D by K array
    # returns a matrix A such that A[i,j] = phi(y_t = i, y_{t+1} = j)
    if (t == 1):
        return np.exp(theta + np.dot(gamma, xl[:, 0]).reshape(D,1) + np.dot(gamma, xl[:,1]).reshape(1,D))
    else:
        return np.exp(theta + np.dot(gamma, xl[:, t-1]).reshape(D,1))

def marginals_and_logPartition(xl, theta, gamma):
    # xl is a K by T matrix of features
    # theta is a D by D array
    # gamma is a D by K array

    m_fwd = np.ones([D, T]) #forward messages. First column is just ones
    # m_fwd[:,t] = m_{t,t+1}
    log_sums = np.zeros(T) #logs of the normalization constants
    for t in range(1,T):
        unnormalized_message = np.dot(phi(t, xl, theta, gamma).T, m_fwd[:, t-1])
        sum_t = sum(unnormalized_message)
        m_fwd[:,t] = unnormalized_message / sum_t
        log_sums[t] = np.log(sum_t)
    
    m_bwd = np.ones([D, T]) #backward messages. Last column is just ones
    # m_bwd[:,t] = m_{t+2,t+1}
    for t in range(T - 2, -1, -1): #T-2,T-3,...,0
        unnormalized_message = np.dot(phi(t+1, xl, theta, gamma), m_bwd[:, t+1])
        m_bwd[:,t] = unnormalized_message / sum(unnormalized_message)

    logPartition = np.sum(log_sums) #log partition function
    p_1 = m_fwd * m_bwd #single node marginals unnormalized
    p_1 = p_1 / p_1.sum(axis=0) #normalized

    #now pairwise marginals
    p_2 = np.zeros([D, D, T-1])
    for t in range(T-1):
        p_2[:,:,t] = phi(t+1, xl, theta, gamma) * m_fwd[:,t].reshape(D,1) * m_bwd[:,t+1].reshape(1,D)# this is the unnormalized marginal on y_{t+1,t+2}
        p_2[:,:,t] = p_2[:,:,t]/p_2[:,:,t].sum() #normalized

    return (p_1,p_2,logPartition)
