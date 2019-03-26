# Study 1.1 : Influence of the number of dimensions

In this study we analyze the influence of the number of dimensions on the average reward / step and on the overall performances of a DDPG agent in our environment.

We first ran this study and found that the more dimensions the better the agent was doing. This could be explained by the fact that if every dimension has a high and low reward on its boundaries, when the environment is reset , there is an increasing chance that the agent is already close to a high reward. To counter this problem we introduced the --reset_radius parameter to contain the starting position in a circle centerd in the environment.

All the training were run several times and then averaged.
For a complete list of the parameters used , please refer to the file study1.1.sh

We ran the DDPG algorithm on the following dimensions, with a consistent repartition of the rewards :

* 2
* 4
* 8
* 16
* 32
* 64
* 128
* 256

Here are  the performances achieved during the whole learning process per dimension :

![all-training](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_dimensions.png)

And a summary of the performances per dimensions :

![summary](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/total_scores.png)

To be able to estimate more accurately the convergence rate and variance of each training batch you will find below the isolated training results.

## 2 Dimensions

![2-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_2.png)

## 4 Dimensions

![4-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_4.png)

## 8 Dimensions
![8-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_8.png)

## 16 Dimensions
![16-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_16.png)

## 32 Dimensions

![32-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_32.png)

## 64 Dimensions

![64-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_64.png)

## 128 Dimensions

![128-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_128.png)

## 256 Dimensions
![256-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_256.png)

## Analysis

As we can see the convergence rate seems to be proportional to the number of dimensions, and as they increase the variance is reduced. A behaviour similar to the one we explained in the introduction might be the explanation.

As DDPG starts evolving in the environment chances are it will reach a high reward on at least one dimension. The gradient towards this dimension will increase so that it can be exploited.

As the process repeats and the number of dimensions increases, The number of dimensions that can be exploited also increases improving the convergence rate, this gives the agent more axis of movement towards a high reward. As there are n / 2 high reward , with an increasing n , the agent expected time to reach a high rewards is stabilized which leads to very low variance.

Above 32 dimensions , we do not notice any significative difference in the performances.
