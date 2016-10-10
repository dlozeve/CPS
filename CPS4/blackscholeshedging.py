import numpy as np
import scipy.stats as stats

def geombrownian(T, mu, sigma, S0, n):
    """Computes a geometric Brownian motion (price of the risky asset).

    """
    dt = 2**(-n) * T
    t = np.linspace(0, T, 2**n)
    W = np.random.standard_normal(size = 2**n)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - sigma**2/2)*t + sigma*W
    return S0 * np.exp(X)

def dplus(s, k, v):
    return np.log(s/k)/np.sqrt(v) + np.sqrt(v)/2

def dminus(s, k, v):
    return np.log(s/k)/np.sqrt(v) - np.sqrt(v)/2

def BS(S0, K, T, sigma, r):
    """Black-Scholes price of a European call option.

    """
    part1 = stats.norm.cdf(dplus(S0, K*np.exp(-r*T), sigma**2*T))
    part2 = stats.norm.cdf(dminus(S0, K*np.exp(-r*T), sigma**2*T))
    return S0*part1 - K*np.exp(-r*T)*part2

def delta(S, K, t, sigma, r):
    """Optimal hedging strategy.

    """
    return stats.norm.cdf(dplus(S, K*np.exp(-r*t), sigma**2*t))

def X(T, n, K, S0, mu, sigma, r):
    res = 0
    t = np.linspace(0, T, 2**n)
    S = geombrownian(T, mu, sigma, S0, n)
    res = np.exp(-r*t)*S
    res = res[1:] - res[:-1]
    deltas = np.zeros(2**n - 1)
    for i in range(2**n-1):
        deltas[i] = delta(S[i], K, T-t[i], sigma, r)
    res = deltas.dot(res)
    # res = delta(S[0], K, T-t[0], sigma, r) *\
    #       np.exp(-r*t[0])*S[0]
    # for i in range(1,2**n):
    #     res += delta(S[i-1], K, T-t[i-1], sigma, r) *\
    #            (np.exp(-r*t[i])*S[i] - np.exp(-r*t[i-1]*S[i-1]))
    res += BS(S0, K, T, sigma, r)
    return np.exp(-r*T) * res


