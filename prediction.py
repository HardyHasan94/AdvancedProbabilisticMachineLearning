"""
In this file the prediction functions are implemented.
How to run:
 - for obtaining the accuracy of the prediction and a data table with the predicted values for each match
   run: accuracy, data = prediction_rate(data, rank, improvement=False) where the arguments
   'data' and 'rank' are obtained from dataProcess().
"""

from gibbsSampler import Gibbs, posteriorDistGaussian
from densityFiltering import dataProcess
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score


def prediction(mu, var, wins, improvement=False):
    """
    This function returns a prediction for a match, by marginalizing out p(t)
    and finding the area of the positive part of the distribution, where t>0.
    If p(t>0) >= 0.5, then player 1 wins, else player 2.

    :param mu: mean of both teams
    :param var: variance of both teams
    :param wins: number of wins for each team so far
    :param improvement: whether to use improvement
    :return: prediction of match result y {1,-1}
    """

    A = np.array([1, -1])  # A matrix in Corollary 2
    mu_t_s = mu[0] - mu[1]  # mean of t given s
    cov_s = np.array([[var[0], 0], [0, var[1]]])
    var_t_s = 2  # variance of t given s

    # variance of t
    var_t = var_t_s + A@cov_s@A[:, np.newaxis]
    mu_t = A@mu[:, np.newaxis]

    # probability of a positive outcome, is the probability of win for player 1
    t_dist = stats.norm(loc=mu_t, scale=np.sqrt(var_t))  # distribution of t
    positive_t = t_dist.cdf(np.infty) - t_dist.cdf(0)  # p(y=1) = p(t > 0)

    # when using improvement
    if improvement:
        # for a player playing in the home city, there is a little higher probability of win, so increase the positive_t
        positive_t += round(np.random.uniform(0.04, 0.08), 2)  # a random number of advantage for home city

        # for two teams, the one with more wins should have a little higher probability of a win
        if wins[0] > wins[1]:
            positive_t += round(np.random.uniform(0.03, 0.10), 2)  # a random number of advantage of wins for team1
        elif wins[0] < wins[1]:
            positive_t -= round(np.random.uniform(0.03, 0.10), 2)  # a random number of advantage of wins for team2

    return 1 if positive_t >= 0.5 else -1  # threshold for win is 50%


def prediction_rate(data, rank, improvement):
    """
    This function implements a prediction mechanism, such that before a match is played,
    a prediction is made based on the mean and variance of the prior skills. If improvement is True,
    then use the improvement difference to boost winning teams.
    It returns the prediction rate and a table with an added column for the predicted result

    :param data: processed data
    :param rank: rank dictionary
    :param improvement: whether to use improvements
    :return: prediction rate, table
    """

    L = 2000  # length of sampling sequence
    burn_in = 5  # use samples after this value

    # add new column prediction to data
    data["prediction"] = np.zeros(shape=(data.shape[0], 1))

    for match in data.iterrows():
        row, team1, team2, y = match[0], match[1].team1, match[1].team2, match[1].y

        prior1, prior2 = rank[team1], rank[team2]

        mu = np.array([prior1[0], prior2[0]])
        var = np.array([prior1[1], prior2[1]])
        wins = np.array([prior1[2], prior2[2]])
        pred = prediction(mu, var, wins, improvement)
        data.loc[row, "prediction"] = pred

        if improvement:
            team1, team2, score1, score2, y = match[1].team1, match[1].team2, match[1].score1, match[1].score2, match[
                1].y
            if y == 1:
                diff_score = score1 - score2
                rank[team1][0] += diff_score  # increment the prior mean skill of team 1 with improvement difference
                rank[team2][0] -= diff_score  # decrement the prior mean skill of team 2 with improvement difference
                rank[team1][2] += 1  # increase wins for team1
            else:
                diff_score = score2 - score1
                rank[team2][0] += diff_score
                rank[team1][0] -= diff_score
                rank[team2][2] += 1  # increase wins for team1

        prior1, prior2 = rank[team1], rank[team2]
        s, _ = Gibbs(L, y, prior1, prior2)  # sample skills using Gibbs sampler
        s = s[burn_in:, :]  # only consider samples after burn-in
        _, _, mu, var = posteriorDistGaussian(s)

        mean1, mean2 = mu.tolist()
        var1, var2 = var.tolist()

        # update rank for the teams
        rank[team1] = [mean1, var1, rank[team1][2]]
        rank[team2] = [mean2, var2, rank[team2][2]]

    accuracy = accuracy_score(data["y"], data["prediction"])
    print(f"The accuracy of this model is: {round(accuracy, 2)}")
    return accuracy, data

def show_prediction_table(data, improvement=False):
    """
    Function that returns the table and accuracy. If improvement is true, then use the improvements.
    :param data: dataset
    :param improvement: whether to use improvement
    :return: accuracy, data table
    """
    data, rank = dataProcess(data, improvement)
    accuracy, data = prediction_rate(data, rank, improvement)
    return accuracy, data


if __name__ == "__main__":
    data = "SerieA.csv"
    improvement = True
    data, rank = dataProcess(data, improvement)
    accuracy, data = prediction_rate(data, rank, improvement)
