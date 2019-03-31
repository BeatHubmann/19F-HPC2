import numpy as np
import matplotlib.pyplot as plt

samples = np.genfromtxt('grass.in', skip_header=1)
plt.scatter(samples[:, 0], samples[:, 1], c=samples[:, 2])
plt.colorbar()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel('X [km]')
plt.ylabel('Y [km]')
plt.title('sampled indicative grass height on grazing grounds')
plt.show()