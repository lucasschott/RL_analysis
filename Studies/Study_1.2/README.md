# Study 1.2 : Influence of the number of dimensions

In this study we analyze the influence of the number of dimensions on the average reward / step and on the overall
performances of a DDPG agent in our environment. On this study the rewards are only on the first dimension and all the other
dimensions are useless to find them.

All the training were run 8 times and then averaged.
For a complete list of the parameters used , please refer to the file study1.2.sh

We ran the DDPG algorithm on the following dimensions :

* 2
* 4
* 8
* 16
* 32
* 64
* 128
* 256

Here are  the performances achieved during the whole learning process per dimension :

![all-training](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_.png)

And a summary of the performances per dimensions :

![summary](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/total_scores.png)

To be able to estimate more accurately the convergence rate and variance of each training batch you will find below the isolated training results.

2 Dimensions | 4 Dimensions
:-------------------:|:-------------------:
![2-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_2.png) | ![4-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_4.png)
8 Dimensions | 16 Dimensions
![8-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_8.png) | ![16-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_16.png)
32 Dimensions | 64 Dimensions
![32-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_32.png) | ![64-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_64.png)
128 Dimensions | 256 Dimensions
![128-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_128.png) | ![256-dimensions](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_256.png)

## Analysis
