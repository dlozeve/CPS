import numpy as np
from scipy import stats
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from browniansimulation import *


mean = np.zeros(12)
var = np.zeros(12)
cov = np.zeros(12)
for n in range(0,12):
    mean[n], var[n], cov[n] = meanvarcov(n, 1000)

plt.plot(mean)
plt.xlabel("n")
plt.ylabel("Mean")
plt.savefig("mean_forward.png")
plt.close()

plt.plot(var)
plt.xlabel("n")
plt.ylabel("Variance")
plt.savefig("var_forward.png")
plt.close()

plt.plot(cov)
plt.xlabel("n")
plt.ylabel("Covariance")
plt.savefig("cov_forward.png")
plt.close()


mean = np.zeros(10)
var = np.zeros(10)
cov = np.zeros(10)
for n in range(0,10):
    mean[n], var[n], cov[n] = backwardmeanvarcov(n, 1000)

plt.plot(mean)
plt.xlabel("n")
plt.ylabel("Mean")
plt.savefig("mean_backward.png")
plt.close()

plt.plot(var)
plt.xlabel("x")
plt.ylabel("Variance")
plt.savefig("var_backward.png")
plt.close()

plt.plot(cov)
plt.xlabel("x")
plt.ylabel("Covariance")
plt.savefig("cov_backward.png")
plt.close()


m = 1000
T = 1

qv_forward = np.zeros(10)
qv_backward = np.zeros(10)

for n in range(10):
    for j in range(m):
        qv_forward[n] += quadraticvariation(brownianmotion(T,n))
    qv_forward[n] /= m

    for j in range(m):
        qv_backward[n] += quadraticvariation(backwardbrownian(T,n))
    qv_backward[n] /= m

plt.plot(qv_forward)
plt.xlabel("n")
plt.ylabel("Quadratic variance (forward simulation)")
plt.savefig("qv_forward.png")
plt.close()

plt.plot(qv_backward)
plt.xlabel("n")
plt.ylabel("Quadratic variance (backward simulation)")
plt.savefig("qv_backward.png")
plt.close()
