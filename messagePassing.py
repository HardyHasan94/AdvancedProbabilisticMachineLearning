"""
In this file, a message passing protocol is implemented.
How to run:
 - in order to obtain a plot comparing the distributions from message passing
   with the ones from gibbs sampling, then use: plot_messageGibbs()
"""

import numpy as np
from scipy.stats import norm, truncnorm
from gibbsSampler import Gibbs, posteriorDistGaussian
import matplotlib.pyplot as plt

def mutiplyGauss(m1, s1, m2, s2):
    # Computes the Gaussian distribution N(m,s) being propotional to N(m1,s1) * N(m2,s2)
    # Input: mean (m1, m2) and variance (s1, s2) of first and second Gaussians, respectively
    # Output: m, s mean and variance of the product Gaussian
    s = 1 / (1/s1 + 1/s2)
    m = (m1/s1 + m2/s2) * s
    return m, s

def divideGauss(m1, s1, m2, s2):
    # Computes the Gaussian distribution N(m,s) being propotional to N(m1,s1) / N(m2,s2)
    # Input: mean (m1, m2) and variance (s1, s2) of first and second Gaussians, respectively
    # Output: m, s mean and variance of the quotient Gaussian
    m, s = mutiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):
    # Computes the mean and variance of a truncated Gaussian distribution
    # Inputs: The interval [a, b] on which the Gaussian (mean mo, var s0) is being truncated
    # Output: m, s mean and variance of the truncated Gaussian
    # scale interval with mean and variance
    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s


def messagePassing():
    s1_mean = 1 # The mean of the prior S1
    s1_var = 0.5 # The variance of the prior S1
    s2_mean = 1 # The mean of the prior S2
    s2_var = 0.5 # The variance of the prior S2
    t_var = 2 # The variance of p(t|S1, S2)
    y0 = 1 # The measurement

    # Computed that mu2(t) does not depend on the hyperparameters -> compute messages from mu3 onwards

    # Message mu3 from prior to node S1
    mu3_mean = s1_mean # mean of message
    mu3_var = s1_var # variance of message

    # Message mu4 from node S1 to factor f_st
    mu4_mean = mu3_mean # mean of message
    mu4_var = mu3_var # variance of message

    # Message mu5 from prior to node S2
    mu5_mean = s2_mean # mean of message
    mu5_var = s2_var # variance of message

    # Message mu4 from node S2 to factor f_st
    mu6_mean = mu5_mean # mean of message
    mu6_var = mu5_var # variance of message

    # Message mu7 from factor f_st to node t
    mu7_mean = mu4_mean - mu6_mean
    mu7_var = mu4_var + mu6_var + t_var

    # Moment matching of t given y
    if y0 == 1:
        a, b = 0, 1000
    else:
        a, b = -1000, 0
    pt_mean, pt_var = truncGaussMM(a, b, mu7_mean, mu7_var)

    # Compute the message from t to f_st
    mu8_mean, mu8_var = divideGauss(pt_mean, pt_var, mu7_mean, mu7_var)

    # Compute the message from f_st to s_1
    mu9_mean = mu6_mean + mu8_mean
    mu9_var = mu6_var + t_var + mu8_var

    # Compute the message from f_st to s_2
    mu10_mean = mu4_mean - mu8_mean
    mu10_var = mu4_var + t_var + mu8_var

    # Compute the posterior of s1 given y
    ps1_mean, ps1_var = mutiplyGauss(mu3_mean, mu3_var, mu9_mean, mu9_var)

    # Compute the posterior of s2 given y
    ps2_mean, ps2_var = mutiplyGauss(mu5_mean, mu5_var, mu10_mean, mu10_var)

    return ps1_mean, ps1_var, ps2_mean, ps2_var


def plot_messageGibbs():
    L = 2000
    s, t = Gibbs(L, y=1)
    ps1_mean, ps1_var, ps2_mean, ps2_var = messagePassing()

    axis = np.linspace(start=-5, stop=5, num=100)
    burn_in = 5
    s = s[burn_in:, :]  # only consider samples after the burn-in. 40 seems to be a good value
    L = s.shape[0]  # sample size
    s1 = s[:, 0]  # skills of player 1
    s2 = s[:, 1]  # skills of player 2
    fitted_gaussian1, fitted_gaussian2, _, _ = posteriorDistGaussian(s)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set(xlabel='s1', ylabel='Fraction', title='Gibbs vs Message passing, both players', xlim=[-5,5])

    ax1.hist(s1, bins=50, color='cyan', density=True, label='Player1', alpha=0.5)
    ax1.plot(np.sort(s1), fitted_gaussian1.pdf(np.sort(s1)), label='Gibbs_P1')
    ax1.plot(axis, norm(ps1_mean, np.sqrt(ps1_var)).pdf(axis), label='messagePassing_P1')

    ax1.hist(s2, bins=50, color='purple', density=True, label='Player2', alpha=0.4)
    ax1.plot(np.sort(s2), fitted_gaussian2.pdf(np.sort(s2)), label='Gibbs_P2')
    ax1.plot(axis, norm(ps2_mean, np.sqrt(ps2_var)).pdf(axis), label='messagePassing_P2')

    plt.legend()
    plt.savefig(f'messagePassing_{L}')
    plt.show()


if __name__ == "__main__":
    plot_messageGibbs()
