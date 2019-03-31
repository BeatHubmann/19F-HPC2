import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncexpon
from scipy.stats import norm


def f_rvs(num_rvs):
    return norm.rvs(size=num_rvs)


def f(x):
    return norm.pdf(x)


def g_rvs(num_rvs, limit=4.5):
    return truncexpon.rvs(loc=limit, b=np.inf, size=num_rvs)


def g(x, limit=4.5):
    return truncexpon.pdf(x, loc=limit, b=np.inf)


def h(x, limit=4.5):
    return np.where(x > limit, 1.0, 0.0)


def I_1(rvs, limit=4.5):
    return np.mean(h(rvs, limit))


def I_2(rvs, limit=4.5):
    return np.mean(h(rvs, limit) * f(rvs) / g(rvs, limit))


N = 10000
cut_off = 4.5

norm_rvs = f_rvs(N)
result_1 = I_1(norm_rvs, cut_off)

expon_rvs = g_rvs(N, cut_off)
result_2 = I_2(expon_rvs, cut_off)

result_true = 1.0 - norm.cdf(cut_off)

print('Monte Carlo integration result = {:.5f}'.format(result_1))
print('Importance sampling integration result = {:.5f}'.format(result_2))
print('True result = {:.5f}'.format(result_true))
print(80 * '-')
print('Monte Carlo integration error = {:.5f}'.format(
    np.abs(result_1-result_true)))
print('Importance sampling integration error = {:.5f}'.format(
    np.abs(result_2-result_true)))

plt.hist(norm_rvs, bins=100,
         label='$Yi$ from standard normal distribution',
         alpha=0.5, density=True)
plt.hist(expon_rvs, bins=100,
         label='$X_i$ from exponential distribution truncated at {}'.format(
             cut_off),
         alpha=0.5, density=True)
plt.plot(np.linspace(-5, 5, 10000), norm.pdf(np.linspace(-5, 5, 10000)),
         label='Standard normal pdf')
plt.legend()
plt.show()
