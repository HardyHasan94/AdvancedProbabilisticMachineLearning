"""
In this file the Assumed Density Filtering is implemented.
How to run:
 - to see the final rankings of the teams, choose one of the datasets
   data = 'SerieA.csv' or data = 'Kabaddi', and run main with show_table(data)
"""

from gibbsSampler import Gibbs, posteriorDistGaussian
from IPython.display import display
import pandas as pd
import numpy as np

def dataProcess(data, improvement):
    """
    This function reads the dataset and process it to remove draw matches, and create a new column for result 'y'.
    If improvement is set to true, then use the improvement too.
    :param data: 'SeriesA.csv' or 'Kabaddi' dataset
    :param improvement: whether to use improvement
    :return: processed dataset, plus a rank dictionary of (team: [mean, var])
    """
    # reading data
    if data == 'SerieA.csv':
        data = pd.read_csv(data)
        # drop all matches with a draw
        data = data.drop(data[data.score1 == data.score2].index)

    elif data == 'Kabaddi':
        data = pd.read_csv('2019_Team Score-Table 1.csv', names=['Week', 'team1', 'team2', 'score1', 'score2',
                                                                   'Winner', 'MarginScore'], header=0)
        data = data.drop(data[data.score1 == data.score2].index)
    # extract score columns
    result = data[["score1", "score2"]]
    # create new column y
    y = result.apply(func=(lambda x: 1 if x[0] > x[1] else -1), axis=1)
    # add new result column to data
    data["y"] = y
    # get a set of all teams
    teams = set(data.team1)

    # for each team, assume the same prior mean and variance, 0 wins
    # team: [mu, var, w] mu=mean, var=variance, where w=wins
    rank = {team: [1, 0.5, 0] for team in teams}

    # if improvement is True, then return the scores too for ranking the teams.
    if improvement:
        return data[["team1", "team2", "score1", "score2", "y"]], rank

    return data[["team1", "team2", "y"]], rank


def ADF(data, rank, improvement):
    """
    This function implements the assumed density filtering, by using Gibbs sampling to compute the posterior
    mean and variance for each match, and then update the mean and variance for the two teams in the rank dict.

    :param data: processed data
    :param rank: rank dictionary
    :param improvement: whether to use the improvements
    :return: final ranking of the teams sorted in descending order
    """

    L = 2000  # length of sampling sequence
    burn_in = 5  # use samples after this value

    for match in data.iterrows():
        if improvement:
            team1, team2, score1, score2, y = match[1].team1, match[1].team2, match[1].score1, match[1].score2, match[
                1].y
            if y == 1:
                diff_score = score1 - score2
                rank[team1][0] += diff_score  # increment the prior mean skill of team 1 with score difference
                rank[team2][0] -= diff_score  # decrement the prior mean skill of team 2 with score difference
                rank[team1][2] += 1  # increase wins for team1
            else:
                diff_score = score2 - score1
                rank[team2][0] += diff_score
                rank[team1][0] -= diff_score
                rank[team2][2] += 1  # increase wins for team1
        else:
            team1, team2, y = match[1].team1, match[1].team2, match[1].y

        prior1, prior2 = rank[team1], rank[team2]

        s, _ = Gibbs(L, y, prior1, prior2)  # sample skills using Gibbs sampler
        s = s[burn_in:, :]  # only consider samples after burn-in
        _, _, mu, var = posteriorDistGaussian(s)
        mean1, mean2 = mu.tolist()
        var1, var2 = var.tolist()

        # update rank for the teams
        rank[team1] = [mean1, var1, rank[team1][2]]
        rank[team2] = [mean2, var2, rank[team2][2]]

    values = rank.values()
    means = [m[0] for m in values]
    variances = [v[1] for v in values]
    wins = [w[2] for w in values]

    rank_table = pd.DataFrame()
    rank_table["Teams"] = rank.keys()
    rank_table["Mean"] = means
    rank_table["Variance"] = variances
    if improvement:
        rank_table["Wins"] = wins
        # sort table
        rank_table.sort_values(by=["Mean", "Wins"], axis=0, ascending=False, inplace=True, ignore_index=True)
        return rank_table
    else:
        rank_table.sort_values(by=["Mean", "Variance"], axis=0, ascending=False, inplace=True, ignore_index=True)

    return rank_table


def show_table(data, randomize=False, improvement=False):
    """
    Function that shows the resulting rank_table. If randomize is true then data is shuffled.
    If improvement is true, then use improvements.

    :param data: dataset
    :param randomize: true/false
    :param improvement: whether to use improvement
    :return: None
    """
    data, rank = dataProcess(data, improvement)
    if randomize:
        data = data.sample(frac = 1)
    table_rank = ADF(data, rank, improvement)
    variance_range = [np.min(table_rank["Variance"]), np.max(table_rank["Variance"])]
    print(f"Range of variances: {variance_range}")
    display(table_rank)


if __name__ == "__main__":
    data = 'SerieA.csv'
    show_table(data, improvement=True)
