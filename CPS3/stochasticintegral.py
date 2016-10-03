import numpy as np
from scipy import stats

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

def I(bm):
    delta = bm[1:] - bm[:-1]
    return delta.dot(bm[:-1])

def J(bm):
    delta = bm[1:] - bm[:-1]
    return delta.dot(bm[1:])

def K(bm):
    delta = bm[1:] - bm[:-1]
    return delta.dot((bm[:-1] + bm[1:])/2)

def halfsquared(bm):
    return bm[-1]**2 * 0.5

def approximationsijk(N, n):
    """Creates a sample of N copies of the random variables 1/2*W_T^2 -
    I_n, 1/2*W_T^2 - J_n, and 1/2*W_T^2 - K_n, and returns their
    sample means.

    """
    # We generate N Brownian motions of length 2**n.
    bms = np.zeros((N,2**n))
    for j in range(N):
        bms[j] = brownianmotion(1,n)
    # Then we compute the quantities I, J, and K.
    isample = np.apply_along_axis(I, 1, bms)
    jsample = np.apply_along_axis(J, 1, bms)
    ksample = np.apply_along_axis(K, 1, bms)
    hssample = np.apply_along_axis(halfsquared, 1, bms)
    isample = hssample - isample
    jsample = hssample - jsample
    ksample = hssample - ksample
    return [isample.mean(), jsample.mean(), ksample.mean()]

def A(bm):
    length = len(bm)
    delta = bm[1:] - bm[:-1]
    expn = np.exp(np.arange(length)/length)
    return delta.dot(expn[:-1])

def B(bm):
    length = len(bm)
    delta = bm[1:] - bm[:-1]
    expn = np.exp(np.arange(length)/length)
    return delta.dot(expn[1:])

def C(bm):
    length = len(bm)
    delta = bm[1:] - bm[:-1]
    expn = np.exp(np.arange(length)/length)
    return delta.dot((expn[:-1]+expn[1:])/2)

def approximationsabc(N, n):
    """Creates a sample of N copies of the random variables 1/2*W_T^2 -
    A_n, 1/2*W_T^2 - B_n, and 1/2*W_T^2 - C_n, and returns their
    sample means.
    
    """
    # We generate N Brownian motions of length 2**n.
    bms = np.zeros((N,2**n))
    for j in range(N):
        bms[j] = brownianmotion(1,n)
    # Then we compute the quantities I, J, and K.
    asample = np.apply_along_axis(A, 1, bms)
    bsample = np.apply_along_axis(B, 1, bms)
    csample = np.apply_along_axis(C, 1, bms)
    hssample = np.apply_along_axis(halfsquared, 1, bms)
    asample = hssample - asample
    bsample = hssample - bsample
    csample = hssample - csample
    return [asample.mean(), bsample.mean(), csample.mean()]

def newa(bm):
    length = len(bm)
    return np.sin(bm[-1]) - np.sum(np.sin(bm[:-1]))/(length*2)

def samplemeana(N, n):
    bms = np.zeros((N,2**n))
    for j in range(N):
        bms[j] = brownianmotion(1,n)
    sample = np.apply_along_axis(newa, 1, bms)
    return sample.mean()

