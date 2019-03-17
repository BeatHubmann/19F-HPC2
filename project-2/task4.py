### HPCSE II Spring 2019
### HW 2 - Task 4: Data Analysis

###############################################################################
### Import Modules
###############################################################################
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
### Subtask 1. Read Data
###############################################################################

data = np.load('task4.npy') # full data set

# data = data[1:100] # Hint: use this line for testing purpose 



###############################################################################
### Subtask 2. Histogram
###############################################################################

def hist(xarr, nbins, continuous=True):
    min_val = xarr.min()
    max_val = xarr.max()
    count   = np.zeros(int(nbins))
    bins    = np.linspace(min_val, max_val, num=nbins)
    for x in xarr:
        bin_number = int((nbins-1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1
   
    """
    TODO:
        Task 2.b - add your changes here (remove this comment)
        I'm assuming you were referring to task 3 here.
    """
    count /= xarr.size # normalize count to get pdf
    assert (np.sum(count) == 1.0), 'pdf not normalized' # assert normalization
    return count, bins

"""
TODO:
    Task 2.a - play with param numbins and study the full data set

"""
numbins = 13 # given data set has 13 different elements
counts, bins = hist(data,numbins) 


###############################################################################
### Subtask 2. Visualise Data
### Subtask 3. Visualise Data
###############################################################################

plt.bar(bins, counts, width=0.5, align='edge', label='data histogram')
# plt.show()  # Hint: you might want to comment this line as you advance with
            # the exercise in order to avoid interuptions



###############################################################################
### Subtask 4. (nothing to do here - only on paper)
###############################################################################



###############################################################################
### Subtask 5. Likelihood and Log-likelihood
###############################################################################

""" 
TODO:
    Subtask 5. - Implement two functions that calculate 
        a) the likelihood
        b) the loglikelihood

    of your distribution function. 

    For a Gaussian distribution this could look like:

    lk_gaussian  = lambda dat, mu, var: (2*np.math.pi*var)**(-0.5*len(dat)) * ...
    llk_gaussian = lambda dat, mu, var: -0.5*len(dat)*np.log(2*np.math.pi) - ...

"""
def lk_poisson(data, theta):
    result = 1
    for k_i in data:
        result *= np.exp(-theta) * theta**k_i / np.math.factorial(k_i)
    return result

def llk_poisson(data, theta): 
    result = -data.size * theta
    for k_i in data:
        result += np.log(theta) * k_i
        result -= np.log(np.math.factorial(k_i))
    return result

###############################################################################
### Subtask 6. Distribution function
###############################################################################

"""
TODO:
    Subtask 6. - Calculate MLE(s) of the params from the data set
    a_hat = ...
    b_hat = ...
    ...

"""
theta_hat = np.average(data)


###############################################################################
### Subtask 7. Comparison with Gaussian Distribution
###############################################################################

"""
TODO:
    Subtask 7. - Calculate the likelihood and log-likelihood given the data and
    your MLE(s). Reuse your functions implemented in Subtask 5.

    For the Gaussian distribution this could look like:
    lik    = lk_gaussian(data, mu_hat, var_hat)
    loglik = llk_gaussian(data, mu_hat, var_hat)

"""
lik_poisson = lk_poisson(data, theta_hat)
loglik_poisson = llk_poisson(data, theta_hat)


###############################################################################
### Subtask 8. Visualisation
###############################################################################

"""
TODO:
    Subtask 8. - Plot the density function of your ditribution and the Gaussian.

    Following the examples of the Gaussian distribution this could look like:
    
    x = np.arange(0, data.max())
    gdensity = list(map(lambda d: lk_gaussian([d], muHat, varHat), x))
    plt.plot(x, gdensity)
    plt.show()

"""
# MLEs for the Gaussian distribution
mu_hat = np.average(data)
var_hat = np.average((data - mu_hat)**2)

# likelihood and loglikelihood for the Gaussian distribution
def lk_gaussian(data, mu, var):
    result = (2 * np.pi * var)**(-len(data) / 2)
    exponent = 0
    for k_i in data:
        exponent += (k_i - mu)**2
    result *= np.exp(-exponent / (2 * var))
    return result   

def llk_gaussian(data, mu, var): 
    result = -len(data) / 2 * np.log(2 * np.pi * var)
    for k_i in data:
        result -= (k_i - mu)**2 / (2 * var)
    return result

# calculating likelihoods for Gaussian distribution
lik_gaussian = lk_gaussian(data, mu_hat, var_hat)
loglik_gaussian = llk_gaussian(data, mu_hat, var_hat)

print('Poisson likelihood = {:.2f}'.format(lik_poisson))
print('Poisson log-likelihood = {:.2f}'.format(loglik_poisson))
print('Gaussian likelihood = {:.2f}'.format(lik_gaussian))
print('Gaussian log-likelihood = {:.2f}'.format(loglik_gaussian))

x = np.arange(0, data.max())
pdensity = list(map(lambda d: lk_poisson([d], theta_hat), x))
gdensity = list(map(lambda d: lk_gaussian([d], mu_hat, var_hat), x))
plt.plot(x, pdensity, label='Poisson distribution, $\Theta$ = {:.2f}'.format(theta_hat))
plt.plot(x, gdensity, label='Gaussian distribution, $(\mu, \sigma^2)$ = ({:.2f}, {:.2f})'.format(mu_hat, var_hat))
plt.legend()
plt.xlabel('$k$')
plt.ylabel('$p(k)$')
plt.show()

