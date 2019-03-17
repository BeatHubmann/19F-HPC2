import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

x_0, gamma = -2.0, 1.0 # given by task

width = 5.0
n_points = 1000
x_range = np.linspace(x_0 - width, x_0 + width, n_points)

cauchy_distr = cauchy(x_0, gamma)
laplace = lambda x, x_0=x_0, gamma=gamma: np.exp(-(x - x_0)**2 / gamma**2) / np.pi / gamma

plt.plot(x_range, cauchy_distr.pdf(x_range),
 label='Cauchy distribution, $(x_0, \gamma)=$ ({:.1f},{:.1f})'.format(x_0, gamma))
plt.plot(x_range, laplace(x_range),
 label='Laplace approximation, $(x_0, \gamma)=$ ({:.1f},{:.1f})'.format(x_0, gamma))
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend()
plt.show()