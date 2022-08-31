# AdvancedProbabilisticMachineLearning

## Introduction
This repository contains the code for the project in the Advanced Probabilistic Machine Learning course at 
Uppsala University. Python 3.7.8 is used for all coding tasks.

The project is a ranking prediction problem, where the task is to rank the teams of the Italian SerieA football league 
based on one season's data. 

## Abstract of our paper
In this project, a Bayesian model is implemented based on the TrueSkill™ Ranking System. Conditional distributions 
and marginal probabilities are computed using the model while different methods for inference 
(i.e., sampling and message passing) are implemented as algorithms. Its evaluation is performed on the Italian soccer 
league, and an extension idea is implemented to improve performance. The average accuracy achieved for the prediction 
of the matches’ result is 0.66.


## Structure
- gibbsSampler.py | the Gibbs sampling algorithm is implemented.
- densityFiltering.py | the Assumed Density Filtering is implemented.
- prediction.py | the prediction functions are implemented.
- messagePassing.py | a message passing protocol is implemented.
- runme.ipynb | cells with all the computations, plots and tables are executed.
  In order to reproduce all the results, it's enough to choose 'Restart & Run All'.

## License
MIT
