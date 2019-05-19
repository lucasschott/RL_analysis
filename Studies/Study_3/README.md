Study 3: Number of dimensions of the environment
----------------------------------------------------

In this study we analyze the influence of the number of dimensions in
the environment on the overall performances of DDPG.

### High and low rewards on each dimension

We test the hypothesis which would be that the more dimensions there are
in the environment more it is difficult for a DDPG agent to learn
something form the environment. This test runs in environments with high
and low rewards on each dimensions, as in Figure \[fig:1d\_env\] and
Figure \[fig:2d\_2\_env\], from environment with 1 dimensions to
environment with 4096 dimensions.

Parameters for this experiment:

-   `--dimensions=` $2^0, 2^2, 2^4, 2^6, 2^8, 2^{10}, 2^{12}, 2^{14}$

-   `--learning_timesteps=5k`

-   `--eval_freq=250`

-   `--exploration_timesteps=32k`

-   `--exploration_mode="uniform"`

-   `--buffer_size=32k`

-   `--speed_limit_mode="vector_norm"`

![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_norm"}](../../report/Study_3/half_norm/scores_dimensions.png "fig:"){width="\linewidth"} | ![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_norm"}](../../report/Study_3/half_norm/total_scores.png "fig:")
|-----------|--------------|

The results shown in Figure \[fig:curves\_dimensions\_half\_norm\] do
not enable us to validate our hypothesis because the agent seems not to
have learned anything for any number of dimensions, it just seems to be
much more efficient with high dimensional environments. Surprisingly the
score increases with the number of dimensions. After reflection it seems
like it can be explained by the structure of the environment and the way
the agent is initially located in it. At the beginning of each epoch the
agent appears at a random position in the environment on each
dimensions, so the more dimensions there are the more likely the agent
will appear near at least one face of the hypercube, such way that it
has to move just few steps to reach the reward. When the number of
dimensions tends to infinity the agent always appears at a distance of
one step from at least one high reward which explains the result of
Figure \[fig:curves\_dimensions\_half\_norm\] which shows that the
average reward per step tends to one when the number of dimensions tends
to infinity.

### High and low rewards on each dimension with the respawn position centered in the environment

In order verify this interpretation we decided conduct a counter
experiment where the agent appears only in the center of the
environment. The goal of this experiment is to make sure that increasing
the number of dimension will not lead to increase the probability for
the agent to appear near an edge. And to really see the influence of the
number of dimension of the learning ability of DDPG. So in this
experiment we run exactly the same experiment as before except that we
introduced a parameter to fix the starting position at the center of the
environment.

Parameters for this experiment:

-   `--dimensions=` $2^0, 2^2, 2^4, 2^6, 2^8, 2^{10}$

-   `--learning_timesteps=5k`

-   `--eval_freq=250`

-   `--exploration_timesteps=32k`

-   `--exploration_mode="uniform"`

-   `--buffer_size=32k`

-   `--speed_limit_mode="vector_norm"`

-   `--reset_radius=0`

![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_reset_norm"}](../../report/Study_3/half_reset_norm/scores_dimensions.png "fig:") | ![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_reset_norm"}](../../report/Study_3/half_reset_norm/total_scores.png "fig:")
|-----------|------------|

Now the result shown in Figure
\[fig:curves\_dimensions\_half\_reset\_norm\] is slightly different, we
can see that for each number of dimensions the agent starts with
approximately the same score. It seems like our interpretation of the
previous experiment was probably true because our modification in the
parameters had the desired effect. We see here that when the number or
dimensions increases the scores reduces. That’s what we expected a the
beginning of this study, and it is probably because the agent starting
for the center of the environment try to move in all the dimensions, and
since its velocity vector norm is limited, if it move in all the
dimensions simultaneously it will move very slowly on each dimension and
then never found any reward which are at the extremities of the
environment on each dimension.

### High and low rewards on each dimension with the respawn position centered in the environment and independent actions

In order to verify if our previous interpretation is right we run the
same test as before, except that now the actions are independent the one
from each other, meaning that if the agent move in all directions at the
same time it will move on each direction at the speed it want, its
velocity vector norm is not bounded the agent just has a max velocity on
each direction. With this modification if our prediction are right, the
performance of DDPG should not decrease while the number of dimensions
increases, because by moving on each dimension as if the environment has
a single dimension, the agent would find the rewards as easily.

Parameters for this experiment:

-   `--dimensions=` $2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7$

-   `--learning_timesteps=5k`

-   `--eval_freq=250`

-   `--exploration_timesteps=32k`

-   `--exploration_mode="uniform"`

-   `--buffer_size=32k`

-   `--speed_limit_mode="independent"`

-   `--reset_radius=0`

![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_reset_indept"}](../../report/Study_3/half_reset_indept/scores_dimensions.png "fig:") | ![Scores according to the number of dimensions with half low rewards and half high rewards[]{data-label="fig:curves_dimensions_half_reset_indept"}](../../report/Study_3/half_reset_indept/total_scores.png "fig:")
|----------|------------|

In Figure \[fig:curves\_dimensions\_half\_reset\_indept\] we can see
that the convergence rate seems to increase with the number of
dimensions, and the standard deviation decreases. This can be explained
by the fact that the agent is now free to explore all the dimensions
independently, the more dimensions there are faster the agent will reach
a high reward on at least one dimension. The gradient towards this
dimension will increase so that it can be exploited. As the process
repeats and the number of dimensions increases, the number of dimensions
that can be exploited also increases improving the convergence rate,
this gives the agent more axis of movement towards a high reward. As
there are $n/2$ high reward, with an increasing n, the agent expected
time to reach a high rewards is stabilized which leads to very low
standard deviation.

We saw that seems to converge for each number of dimensions in this
configuration, so we want now to compare the convergence time needed to
reach the optimal policy according to the number of dimensions. The
agent appears at the center of the environment which is at a minimum
distance of 1 form the edges of the hypercube and the maximum velocity
of the agent is set to 0.1 on each dimension, so if the agent learns the
optimal policy its score would be of 1 reward each 10 steps, which means
a reward per step of 0.1. To compare the convergence of DDPG on the
different size on environment we consider that a policy is optimal when
the agent reach a an average score of $0.8$ rewards per step.

![Convergence time according to the number of dimensions[]{data-label="fig:convergence"}](../../report/Study_3/half_convergence_reset_indept/convergences.png)

Figure \[fig:convergence\] shows that the average timesteps needed to
reach the optimal policy seems to decrease linearly while the number of
dimensions increases.

### High and low rewards on the first dimension only, with the respawn position centered in the environment and independent actions

This study is the same as the previous one, except that we put only one
high and low reward on the first dimension, and all the other dimensions
does not contains any rewards. For example this two environments Figure
\[fig:1d\_env\] and Figure \[fig:2d\_1\_env\] respects this constraint.
We decided to do this study to verify if the result of previous study is
due to the fact that the number of rewards increases with the number of
dimensions. We will see if a lot of dimensions and only rewards on the
first dimension makes the learning harder or not.

Parameters for this experiment:

-   `--dimensions=` $2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7$

-   `--learning_timesteps=5k`

-   `--eval_freq=250`

-   `--exploration_timesteps=32k`

-   `--exploration_mode="uniform"`

-   `--buffer_size=32k`

-   `--speed_limit_mode="independent"`

-   `--reset_radius=0`

-   `--high_reward_count="one"`

-   `--low_reward_count="one"`

![Scores according to the number of dimensions with one low reward and one highreward[]{data-label="fig:curves_dimensions_one"}](../../report/Study_3/one_reset_indept/scores_dimensions.png "fig:") | ![Scores according to the number of dimensions with one low reward and one high reward[]{data-label="fig:curves_dimensions_one"}](../../report/Study_3/one_reset_indept/total_scores.png "fig:")
|----------|---------|

Figure \[fig:curves\_dimensions\_one\] shows the performances achieved
during the whole learning process per dimension. As we can see, the
convergence rate seems to stay stable with the number of dimensions, but
with a very high standard deviation. So it seems to mean that the number
of dimensions does not affect the ability of DDPG to learn the policy on
the important dimension. Since all the dimensions are independents, DDPG
can rely only on the single rewarding dimension in order to achieve the
highest rewarding goal.
