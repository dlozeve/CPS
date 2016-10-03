import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from stochasticintegral import *


N = 1000


res = np.zeros((10, 3))
for n in range(10):
    res[n] = approximationsijk(N, n+10)
res = res.transpose()

plt.plot(res[0])
plt.figsave("imean.png")
plt.close()

plt.plot(res[1])
plt.figsave("jmean.png")
plt.close()

plt.plot(res[2])
plt.figsave("kmean.png")
plt.close()

# I and J converge, but with a bias of +/- 1/2. K converges to 0 very
# quickly. The results are similar to what would be expected for a
# Riemann integral.

res = np.zeros((10, 3))
for n in range(10):
    res[n] = approximationsabc(N, n+10)
res = res.transpose()

plt.plot(res[0])
plt.figsave("amean.png")
plt.close()

plt.plot(res[1])
plt.figsave("bmean.png")
plt.close()

plt.plot(res[2])
plt.figsave("cmean.png")
plt.close()

res = np.zeros((10,3))
for n in range(10):
    res[n] = samplemeana(N,n+10)

plt.plot(res)
plt.figsave("agraph.png")
plt.close()
