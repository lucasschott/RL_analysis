# Study 3 : DDPG behaviour on filtered replay buffer

This study focuses on the behaviour of DDPG with a filtered replay buffer.

The flag --no-new-exp is set for the whole study , this forbids DDPG to add any experiences to the replay buffer.

The environment being 2 dimensional , we can visualize the content of a replay buffer to see the explored regions.

Non filtered replay buffer           |  Filtered replay buffer
:-------------------------:|:-------------------------:
![non-filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/non-filtered.png)   |  ![filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/filtered.png)


An optimal policy can be visualized either as a contour graph or gradient field.

We will compare the results to those obtained by an unfiltered DDPG algorithm allowed to insert new samples in the replay buffer. The high reward are the edges on the top and top-right of the 2 dimensional environment.

Optimal contour           |  Optimal gradient
:-------------------------:|:-------------------------:
![contour](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/contour.png)   |  ![gradient](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/gradient.png)


# Center filter

For this test , the filter is in the center with a radius of 0.4.

With a replay buffer sufficiently large we expect an interpolation of the blind area to be good enough to let DDPG evolve correctly in the environment.

## Impact of replay buffer size on the learned policy 

The number of learning steps is set to 10 000

| Replay buffer size  |  Contour | Gradient  |
|:-:|:-:|:-:|
|  250 |   |   |
|  500 |   |   |
|  1000 |   |   |
|  2500 |   |   |
|  5000 |   |   |
|  10000 |   |   |

## Impact of the number of learning step on the learned policy

The replay buffer size is set to 10 000, and the exploration phase runs untils it is filled.

| Learning steps  |  Contour | Gradient  |
|:-:|:-:|:-:|
|  250 |   |   |
|  500 |   |   |
|  1000 |   |   |
|  2500 |   |   |
|  5000 |   |   |
|  10000 |   |   | 

# Edge filter

For this test , the filter is in the upper right corner and has a 0.6 radius.

|  Replay buffer |
|:-:|
| ![edge-filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/edge-filtered.png)  |
Here , no interpolation is really possible,  we explore how the policy evolves in and around of the blind zone.

## Impact of replay buffer size on the learned policy 

| Replay buffer size  |  Contour | Gradient  |
|:-:|:-:|:-:|
|  250 |   |   |
|  500 |   |   |
|  1000 |   |   |
|  2500 |   |   |
|  5000 |   |   |
|  10000 |   |   |

## Impact of the number of learning step on the learned policy

| Learning steps  |  Contour | Gradient  |
|:-:|:-:|:-:|
|  250 |   |   |
|  500 |   |   |
|  1000 |   |   |
|  2500 |   |   |
|  5000 |   |   |
|  10000 |   |   |
