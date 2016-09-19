# MAP552 - CPS1
# The Cox-Ross-Rubinstein model
# Dimitri Lozeve

# Time spent: 3h

from coxrossrubinstein import *
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Parameters definition
S0 = 100
T = 1
n = 50
r = 0.05
b = 0.05
sigma = 0.3

# Call price with respect to K

K = np.arange(80,121,1)

callprices = np.zeros(41)
for i in range(0,41):
    callprices[i] = Calln(S0, T, n, r, b, sigma, K[i])

plt.plot(K, callprices, lw=1.5)
plt.xlabel('strike price')
plt.ylabel('call price')
plt.savefig("callprices.png")
plt.close()

# The call price decreases when K rises. This is expected behavior:
# when the strike price is low, you can make huge profits at maturity.


# Hedging with respect to K

K = np.arange(80,121,1)
thetas = np.zeros(41)
for i in range(0,41):
    thetas[i] = Deltan(S0, T, n, r, b, sigma, K[i],1)[0]
    
plt.plot(K, thetas, lw=1.5)
plt.xlabel('strike price')
plt.ylabel('call hedging strategy')
plt.savefig("hedging.png")
plt.close()


# Comparison to Black-Scholes

K = 105
errs = np.zeros(100)
for n in range(1,100):
    errs[n] = err(S0, T, n, r, b, sigma, K)

plt.plot(np.arange(0,100,1), errs, 'r-', lw=1.5)
plt.ylabel('error')
plt.xlabel('strike price')
plt.savefig("error.png")
plt.close()

# The error tends to zero when the number of periods rise. Thus,
# Black-Scholes could be seen as the continuous-time limit of
# Cox-Ross-Rubinstein.
