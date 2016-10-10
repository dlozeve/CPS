import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import blackscholeshedging

##################################################

T = 1
sigma = 0.3
mus = [0.05, 0.02, 0.45]
S0 = 100
n = 12

N = 1000

mean = [0,0,0]
var = [0,0,0]
for i in range(len(mus)):
    Ws = np.zeros((N,2**n))
    for j in range(N):
        Ws[j] = geombrownian(T, mus[i], sigma, S0, n)
    mean[i] = np.mean(Ws)
    var[i] = np.var(Ws)

##################################################

## Serial version
## DO NOT RUN

# T = 1
# n = 10
# S0 = 100
# mus = [0.05, 0.02, 0.45]
# sigma = 0.3
# r = 0.05
# N = 1000

# Xs = np.zeros((3,40,N))

# for i in range(1):
#     mu = mus[i]
#     for j in range(20):
#         K = 100 + j
#         for k in range(N):
#             Xs[i,j,k] = X(T, n, K, S0, mu, sigma, r)
# Xs

##################################################

## Parallel version

from ipyparallel import Client
rc = Client()
dv = rc[:]

with dv.sync_imports():
    import numpy
    import scipy
    import scipy.stats

%autopx
    
def geombrownian(T, mu, sigma, S0, n):
    dt = 2**(-n) * T
    t = numpy.linspace(0, T, 2**n)
    W = numpy.random.standard_normal(size = 2**n)
    W = numpy.cumsum(W) * numpy.sqrt(dt)
    X = (mu - sigma**2/2)*t + sigma*W
    return S0 * numpy.exp(X)

def dplus(s, k, v):
    return numpy.log(s/k)/numpy.sqrt(v) + numpy.sqrt(v)/2

def dminus(s, k, v):
    return numpy.log(s/k)/numpy.sqrt(v) - numpy.sqrt(v)/2

def BS(S0, K, T, sigma, r):
    part1 = scipy.stats.norm.cdf(dplus(S0, K*numpy.exp(-r*T), sigma**2*T))
    part2 = scipy.stats.norm.cdf(dminus(S0, K*numpy.exp(-r*T), sigma**2*T))
    return S0*part1 - K*numpy.exp(-r*T)*part2

def delta(S, K, t, sigma, r):
    return scipy.stats.norm.cdf(dplus(S, K*numpy.exp(-r*t), sigma**2*t))

def X(T, n, K, S0, mu, sigma, r):
    res = 0
    t = numpy.linspace(0, T, 2**n)
    S = geombrownian(T, mu, sigma, S0, n)
    res = numpy.exp(-r*t)*S
    res = res[1:] - res[:-1]
    deltas = numpy.zeros(2**n - 1)
    for i in range(2**n-1):
        deltas[i] = delta(S[i], K, T-t[i], sigma, r)
    res = deltas.dot(res)
    # res = delta(S[0], K, T-t[0], sigma, r) *\
    #       numpy.exp(-r*t[0])*S[0]
    # for i in range(1,2**n):
    #     res += delta(S[i-1], K, T-t[i-1], sigma, r) *\
    #            (numpy.exp(-r*t[i])*S[i] - numpy.exp(-r*t[i-1]*S[i-1]))
    res += BS(S0, K, T, sigma, r)
    return numpy.exp(-r*T) * res


T = 1
n = 10
S0 = 100
sigma = 0.3
r = 0.05
N = 1000

%autopx

%px mu = 0.05
N = 1000
dv.scatter('mu1', numpy.repeat(numpy.arange(100-20,100+20+1), N))
#dv.scatter('mu1', numpy.ones(N)*80)
%px xs1 = [X(T, n, K, S0, mu, sigma, r) for K in mu1]
xs1 = dv.gather('xs1')

numpy.array(xs1.get())

%px mu = 0.02
N = 1000
dv.scatter('mu2', numpy.repeat(numpy.arange(100-20,100+20+1), N))
%px xs2 = [X(T, n, K, S0, mu, sigma, r) for K in mu2]
xs2 = dv.gather('xs2')

numpy.array(xs2.get())

%px mu = 0.45
N = 1000
dv.scatter('mu3', numpy.repeat(numpy.arange(100-20,100+20+1), N))
%px xs3 = [X(T, n, K, S0, mu, sigma, r) for K in mu3]
xs3 = dv.gather('xs3')

numpy.array(xs3.get())

Xs = np.array([np.array(xs1.get()), np.array(xs2.get()), np.array(xs3.get())])

