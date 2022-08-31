"""
In this file the Gibbs sampling is implemented.
How to run:
 - if plotting the samples, then choose an L, y, and run the main with plot_skills(L, s, t), where s,t = Gibbs(L, y=)
 - if plotting the histogram of the skills and the fitted Gaussian, then use plot_histogram(s)
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def conditional_outcome(mu_t, var_t, a, b):
    """
    Function that implements the truncated-normal dist. of the likelihood t given skills s

    :param mu_t: mean of t
    :param var_t: variance of t
    :param a: bound for truncated normal
    :param b: upper bound for truncated normal
    :return: a sample from the truncated normal distribution.
    """
    if a == 0:
        a, b = (a-mu_t)/np.sqrt(var_t), b
    else:
        a, b = a, (b-mu_t)/np.sqrt(var_t)

    return stats.truncnorm.rvs(a, b, loc=mu_t, scale=np.sqrt(var_t))


def posterior(mu_s, cov_s):
    """
    This function returns the posterior distribution of the skills given an outcome.

    :param mu_s: mean of player skills
    :param cov_s: covariance of player skills
    :return: a multivariate normal distribution
    """

    return np.random.multivariate_normal(mu_s, cov_s)


def variance(A, cov_s, var_t):
    """
    This function computes the covariance matrix for the posterior distribution.

    :param A: a matrix
    :param cov_s: covariance of player skills
    :param var_t: outcome variance
    :return: covariance matrix
    """

    return np.linalg.inv(np.linalg.inv(cov_s) + A[:, np.newaxis] * (1 / var_t) * A)


def mean(mu_s, A, cov_s, var_t, t):
    """
    This function computes the mean vector for the posterior distribution.

    :param mu_s: prior mean vector
    :param A: a matrix
    :param cov_s: covariance of player skills
    :param var_t: outcome variance
    :param t: match outcome
    :return: mean vector
    """

    cov = variance(A, cov_s, var_t)

    return cov @ (np.linalg.inv(cov_s) @ mu_s[:, np.newaxis] + A[:, np.newaxis] * (t / var_t))


def Gibbs(L, y, prior1=None, prior2=None):
    """
    This function implements a Gibbs sampler to estimate the
    posterior skills distribution given a match result.

    :param L: the sample length
    :param y: match result {1, -1}
    :param prior1: array containing prior mean and variance of player 1
    :param prior2: array containing prior mean and variance of player 2
    :return: sampled posterior skills 's', sampled match outcomes 't'
    """

    # if prior distributions exist, then use them, else assume random but same mean and variance for both players
    if prior1 and prior2:
        mu_1, var_1, _ = prior1
        mu_2, var_2, _ = prior2
    else:
        mu_1, var_1 = 1, 0.5
        mu_2, var_2 = 1, 0.5

    # matrix A, as used in Corollary 2
    A = np.array([1, -1])

    # the hyperparameters
    mu_s = np.array([mu_1, mu_2])  # assume both players have a prior mean skill of 1
    cov_s = np.zeros(shape=(2, 2))  # and a variance of 0.5
    cov_s[0], cov_s[1] = np.array([var_1, 0]), np.array([0, var_2])
    mu_t = mu_s[0] - mu_s[1]  # computing mean of t
    var_t = 2  # assume this variance for outcome 't'

    # the lower/upper limits of the truncated normal are:
    if y == 1:
        a = 0
        b = np.infty
    else:
        a = -np.infty
        b = 0

    # initialize empty arrays for the sampled values
    t = np.zeros(shape=(L, 1))  # array holding sampled outcome-values
    s = np.zeros(shape=(L, 2))  # matrix holding sampled skills-values, column1 <-> player1, column2 <-> player2

    s[0] = np.reshape(np.random.multivariate_normal(mu_s, cov_s), (2,))  # sample prior skills from a normal dist.
    t[0] = conditional_outcome(mu_t, var_t, a, b)

    for k in range(1, L):
        # compute the mean and cov of the posterior p(s|t)
        cov_s_t = variance(A, cov_s, var_t)  # covariance matrix of p(s|t)
        mu_s_t = mean(mu_s, A, cov_s, var_t, t[k-1])  # mean vector of p(s|t)
        mu_s_t = np.reshape(mu_s_t, (2,))

        # sample skills s
        s_sample = posterior(mu_s_t, cov_s_t)  # the posterior skills
        s[k] = s_sample

        # sample outcome t
        mu_t = mu_s_t[0] - mu_s_t[1]  # updating mean of t
        t[k] = conditional_outcome(mu_t, var_t, a, b)

        # mu_s = mu_s_t  # updating mean skills
        # cov_s = cov_s_t  # updating covariance matrix

    return s, t


def plot_skills(L, s, t):
    """
    Function that plots the sampled skills for each player, and the sampled outcomes
    :param L: the sample length
    :param s: sampled skills
    :param t: sampled outcomes
    :return: sampled posterior skills 's', sampled outcomes 't'
    """
    burn_in = 5
    min1 = np.min(s[:, 0])-4
    max1 = np.max(s[:, 0])+4
    min2 = np.min(s[:, 1])-4
    max2 = np.max(s[:, 1])+4

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ax1.set(xlabel='L', ylabel='skills', title='Sampled skills for Player 1', ylim=[min1, max1])
    ax2.set(xlabel='L', ylabel='skills', title='Sampled skills for Player 2', ylim=[min2, max2])
    ax3.set(xlabel='L', ylabel='outcome', title='Sampled outcome', ylim=[np.min(t)-4, np.max(t)+4])

    ax1.plot(s[:, 0], color='blue')
    ax2.plot(s[:, 1], color='purple')
    ax3.plot(t, color='cyan')
    plt.savefig(f'skills_{L}')
    plt.show()

    return s, t


def posteriorDistGaussian(s):
    """
    Function that approximates Gaussian distributions for the posterior skills given sampled skills s.

    :param s: sampled skills using Gibbs sampler
    :return:  two separate univariate Gaussian fitted Gaussian distributions of each player,
              the mean and variances of both players based on the samples.
    """

    mu = np.mean(s, axis=0)
    var = np.var(s, axis=0)

    return stats.norm(loc=mu[0], scale=np.sqrt(var[0])), stats.norm(loc=mu[1], scale=np.sqrt(var[1])), mu, var


def plot_histogram(s):
    """
    Function that plots the histogram of sampled skills after the burn-in together with a plot of the fitted Gaussian
    for the posterior distribution.

    :param s: array of sampled skills for each player.
    :return: None
    """
    burn_in = 5
    s = s[burn_in:, :]  # only consider samples after the burn-in. 40 seems to be a good value
    L = s.shape[0]  # sample size
    s1 = s[:, 0]  # skills of player 1
    s2 = s[:, 1]  # skills of player 2
    fitted_gaussian1, fitted_gaussian2, _, _ = posteriorDistGaussian(s)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.set(xlabel='s1', ylabel='Fraction', title=f'Normalized Skills histogram for Player 1, L={str(L)}')
    ax2.set(xlabel='s2', ylabel='Fraction', title=f'Normalized Skills histogram for Player 2, L={str(L)}')

    ax1.hist(s1, color='cyan', density=True, label='Hist')
    ax1.plot(np.sort(s1), fitted_gaussian1.pdf(np.sort(s1)), color='green', label='pdf')

    ax2.hist(s2, color='purple', density=True, label='Hist')
    ax2.plot(np.sort(s2), fitted_gaussian2.pdf(np.sort(s2)), color='green', label='pdf')
    plt.legend()
    plt.savefig(f'hist_{L}')
    plt.show()


if __name__ == "__main__":
    # L is the number of samples
    L = 9
    # plot the sampled skills
    s, t = Gibbs(L, y=1)
    plot_skills(L, s, t)

    # plot the histograms with fitted Gaussians
    plot_histogram(s)

