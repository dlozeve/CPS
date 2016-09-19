# MAP552 - CPS1
# The Cox-Ross-Rubinstein model
# Dimitri Lozeve

# Time spent: 3h

import numpy as np
import scipy.stats as stats

def Sn(S0, T, n, b, sigma, j):
    """Returns the risky asset price.

    The parameters used are from the Cox-Ross-Rubinstein model. j is
    the time period.

    """
    # Relevant parameters
    h = T/n # time period
    u = np.exp(b*h + sigma*np.sqrt(h)) # up value
    d = np.exp(b*h - sigma*np.sqrt(h)) #down value
    s = np.zeros(j+1)
    for i in range(0,j):
        # We simply apply the formula for S.
        s[i] = S0 * u**(j-i) * d ** i
    return s

def Payoffn(S0, T, n, b, sigma, K):
    """Returns the payoff of a European call option with maturity T and
    strike K.

    """
    s = Sn(S0, T, n, b, sigma, n)
    return np.maximum(s - K*np.ones(n+1), 0)

def Calln(S0, T, n, r, b, sigma, K):
    """Returns the initial price of the call option.

    We compute it recursively, starting from the payoff, and computing
    the value at the previous time using the recurrence formula with
    the discount rate.

    """
    h = T/n
    u = np.exp(b*h + sigma*np.sqrt(h))
    d = np.exp(b*h - sigma*np.sqrt(h))
    p = (np.exp(r*h) - d) / (u - d) # risk-neutral measure

    a = np.zeros((n+1,n+1))
    # We fill the last line with the payoff
    a[n] = Payoffn(S0, T, n, b, sigma, K)
    j = n-1
    while j >= 0: # going from the last line to the first
        for i in range(0,j+1): # we build the tree from right to left
            a[j,i] = np.exp(-r * h) * (p * a[j+1,i] + (1-p) * a[j+1,i+1])
        j -= 1
    return a[0,0] # the initial value

def Deltan(S0, T, n, r, b, sigma, K, j):
    """Returns the hedging strategy for the call option.

    Same as Calln(...), we compute both values recursively from the
    last one, stopping at time j.

    """
    h = T/n
    u = np.exp(b*h + sigma*np.sqrt(h))
    d = np.exp(b*h - sigma*np.sqrt(h))
    p = (np.exp(r*h) - d) / (u - d)

    B = np.zeros((n+1,n+1))
    theta = np.zeros((n+1,n+1))
    B[n] = Payoffn(S0, T, n, b, sigma, K)
    k = n-1
    while k >= j:
        for i in range(0,j):
            B[k,i] = np.exp(-r * h) * (p * B[k+1,i] + (1-p) * B[k+1,i+1])
            theta[k,i] = (B[k+1,i+1] - B[k+1,i]) / (Sn(S0, T, n, b, sigma, k)[i] * (u-d))
        k -= 1
    return theta[j]

def Call(S0, T, r, sigma, K):
    """Returns the initial Balck-Scholes price of the call option.

    Direct application of the formula involving the distribution
    function of the gaussian.

    """
    d1 = 1/(sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    return stats.norm.cdf(d1)*S0 - stats.norm.cdf(d2)*K*np.exp(-r*T)

def err(S0, T, n, r, b, sigma, K):
    """Returns the error, i.e. the difference between the call price in
    the Cox-Ross-Rubinstein model and the Black-Scholes model.

    """
    return Calln(S0,T,n,r,b,sigma,K)/Call(S0,T,r,sigma,K) -1

