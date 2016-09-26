
import numpy as np
from scipy import stats

# Forward simulation

def brownianmotion(T,n):
    """Returns a Brownian motion of size 2**n, with total time T.
    
    """
    # standard normal distributed random variables Z_i
    z = stats.norm.rvs(size=2**n)
    w = np.zeros(2**n)
    var = np.sqrt(2**(-n) * T)
    for i in range(1,2**n):
        w[i] = w[i-1] + z[i]*var
    return w

def meanvarcov(n, m):
    """Returns the sample mean and variance of W_T and covariance of
    W_T/2, W_T, using a Brownian motion of size 2**n and a sample of
    size m.

    """
    res = np.zeros((m,2**n))
    for j in range(0,m):
        res[j] = brownianmotion(1,n)
    mean = np.mean(res[:,-1])
    var = np.var(res[:,-1])
    cov = np.cov(res[:,2**n/2], res[:,-1])[0,1]
    return mean, var, cov


# Backward simulation

def completebrownian(a, dt):
    """Completes a brownian motion, given its to extremities and the time
    difference.

    """
    n = len(a)
    if n > 2:
        loc = (a[0]+a[-1])/2
        scale = dt/4
        a[n/2] = stats.norm.rvs(loc = loc, scale = scale)
        completebrownian(a[:n/2+1],dt/2)
        completebrownian(a[n/2:],dt/2)

def backwardbrownian(T, n):
    """Returns a Brownian motion of size 2**n, using backward simulation.

    """
    res = np.zeros(2**n)
    res[-1] = stats.norm.rvs(scale=T)
    completebrownian(res,T)
    return res

def backwardmeanvarcov(n, m):
    """Returns the sample mean and variance of W_T and covariance of
    W_T/2, W_T, using a Brownian motion of size 2**n and a sample of
    size m.

    """
    res = np.zeros((m,2**n))
    for j in range(0,m):
        res[j] = backwardbrownian(1,n)
    mean = np.mean(res[:,-1])
    var = np.var(res[:,-1])
    cov = np.cov(res[:,2**n/2], res[:,-1])[0,1]
    return mean, var, cov



# Quadratic variation

def quadraticvariation(bm):
    """Computes the quadratic variation of a given Brownian motion on its
    whole life.

    """
    m = len(bm)
    res = 0
    for i in range(m):
        res += (bm[i] - bm[i-1])**2
    return res

