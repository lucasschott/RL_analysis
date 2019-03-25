# Study 3 : DDPG behaviour on filtered replay buffer

This study focuses on the behaviour of DDPG with a filtered replay buffer.

Non filtered replay buffer           |  Filtered replay buffer
:-------------------------:|:-------------------------:
![non-filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/non-filtered.png)   |  ![filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/filtered.png)


An optimal policy can be visualized either as a contour graph or gradient field.

We will compare the results to those obtained by an unfiltered DDPG algorithm allowed to insert new samples in the replay buffer. The high reward being places on the top and right edge of the 2 dimensional environment.

Optimal contour           |  Optimal gradient
:-------------------------:|:-------------------------:
![contour](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/contour.png)   |  ![gradient](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/gradient.png)


# Center filter

For this test , the filter is in the center with a radius of 0.4.

With a replay buffer sufficiently large we expect an interpolation of the blind area to be good enough to let DDPG evolve correctly in the environment.

## Impact of replay buffer size on the learned policy 

## Impact of the number of learning step on the learned policy

# Edge filter

For this test , the filter is in the upper right corner and has a 0.4 radius.

Here , no interpolation is really possible,  we explore how the policy evolved in and around of the blind zone.

## Impact of replay buffer size on the learned policy 

## Impact of the number of learning step on the learned policy

