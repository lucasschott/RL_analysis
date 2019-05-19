Study 1: Size of the replay buffer
----------------------------------

According to [@fujimoto_off-policy_2018] a drawback of DDPG is that it
has trouble learning from a fixed batch of experiences in a replay
buffer without having the possibility to add new experiences in this
replay buffer. So in this study we analyze the influence of the size of
a fixed batch of experiences in a replay buffer on the average score of
DDPG.

This study runs with the batch learning method, this means that the
replay buffer is filled before the beginning of the training and DDPG is
not allowed to push any new experience into the replay buffer, making
the exploration phase the only source of information on the environment.
So it is important to choose how to proceed to this exploration. We
decided to test the two common ways to do this, the first one, uniform
exploration consists in drawing an initial state from a uniform
distribution, then an action is sampled from another uniform
distribution over the action space. The full experience is then created
when applying the sampled action on the initial state. At each
iteration, the whole process restarts. The second exploration scheme is
a random walk where an initial state and an action are sampled as in the
uniform exploration. However, any subsequent step will have the previous
resulting state in place of its initial state.

For this study we train DDPG for 100k timesteps, with a replay buffer
size ranging from 2^4 to 2^16. The environment is 2
dimensional with high rewards present on the top and right edges and low
rewards on the opposite sides.

We expect to see poor performance on the lower replay buffers sizes, as
the number of experiences that can be contained in the replay buffer is
too low, leading to an almost certain auto correlation of the
experiences when sampling from the replay buffer. As the replay buffer
size increases, the performance should rise and settle to a level
dependent upon the environment. In our 2D environment, the starting
position greatly influences the outcome and therefore the average reward
obtained. We forced the starting point to be uniformly generated around
the center point of the environment in a hyper sphere with a radius of
0.1 in order to reduce the variance that would otherwise be induced.

### Uniform sampled replay buffer

![Uniform sampled replay buffers[]{data-label="fig:buffers_uniform"}](../../report/Study_1/buffer/uniform_16.png "fig:") | ![Uniform sampled replay buffers[]{data-label="fig:buffers_uniform"}](../../report/Study_1/buffer/uniform_256.png "fig:") | ![Uniform sampled replay buffers[]{data-label="fig:buffers_uniform"}](../../report/Study_1/buffer/uniform_4096.png "fig:")
|-------------|-------------|--------------|

Figure \[fig:buffers\_uniform\] represents replay buffers examples of
our 2D environment \[fig:2d\_2\_env\]. The visualisations shows the
initial position for each experience contained, and the color from red
to green means from the oldest to newest experience, this information is
not important for a replay buffer filled by uniform exploration, however
for replay buffers filled with random walk it allow us to visualize the
path taken by the agent during the exploration phase.

Parameters for this experiment:

-   `--learning_timesteps=100k`

-   `--eval_freq=5k`

-   `--exploration_mode="uniform"`

-   `--buffer_size=` 2^3, 2^4, 2^6, 2^8, 2^10, 2^12, 2^14

-   `--exploration_timesteps=`
    2^3, 2^4, 2^6, 2^8, 2^10, 2^12, 2^14

![Uniform sampling[]{data-label="fig:scores_uniform"}](../../report/Study_1/1.1/curves1_1.png "fig:") | ![Uniform sampling[]{data-label="fig:scores_uniform"}](../../report/Study_1/1.1/total_scores1_1.png "fig:")
|----------------|--------------|

Figure \[fig:scores\_uniform\] shows the score reached by DDPG for each
replay buffer size, as we can see, the score seems to be converging
quite early. With replay buffers of size 256 and more, 100k learning
steps are enough to converge close to the optimal policy. We can observe
that bigger replay buffers does not necessarily imply better scoring
performances. It can be explained by the uniform character of the
sampling. However even if DDPG can converge with small replay buffers,
we can also observe that the standard deviation reduces when the size of
the replay buffer increases. It can be because with too few examples a
uniform sampling cannot draw a set of positions that is enough
representative of the environment. For example in Figure
\[fig:buffers\_uniform\] (a) we can see that with a replay buffer of
size 16, there was no experience in the top left corner.

![Optimal Actor Critic output for a 64k replay buffer[]{data-label="fig:64k_policy"}](../../report/Study_1/policy/Pi_arrow_65536.png "fig:") | ![Optimal Actor Critic output for a 64k replay buffer[]{data-label="fig:64k_policy"}](../../report/Study_1/policy/Q_contour_65536.png "fig:")
|---------------|--------------|

Figure \[fig:64k\_policy\] show us the optimal policy and the associated
action value learned by DDPG after 100k learning timesteps on a 64k
replay buffer. We can see that the policy and the action value learned
by DDPG are different from the theoretical ones
\[fig:theoretical\_policy\]. Even if the policy is similar to the
theoretical policy by going to the top right corner, we can see that
DDPG did not learn that going straight towards the closest edge will
allow to get the reward quicker that moving in diagonal. We can also
notice that the action value learned by DDPG is a kind of linear
regression of the action value calculated by value iteration in the
discretized environment, and the policy learned by DDPG seems to follow
this linear regression, that’s probably why it move to the top right
corner diagonally.

Trying to investigate on this difference we tried several set of
parameters with learning timesteps ranging from 10K to 1M and changing
the activation functions of the actor and critic networks from ReLu to
Tanh. More can be done but our results tend to show that DDPG is unable
to correctly approximate the real Q value function.

### Random walk explored replay buffer

We saw with the previous study that even with a very small replay
buffer, DDPG can converge in our 2D environment \[fig:2d\_2\_env\]. And
this is probably possible because the environment is very basic and
because the sampling of the replay buffer is uniform. In this study we
will look at whether DDPG can converge on our 2D environment with small
replay buffer, when replay buffer aren’t sampled uniformly but filled
with a random walk exploration. So the parameters of the experiment are
the sames as the previous one, except how to fill the buffer.

Parameters for this experiment:

-   `--learning_timesteps=100k`

-   `--eval_freq=5k`

-   `--exploration_mode="random_walk"`

-   `--buffer_size=` 2^3, 2^4, 2^6, 2^8, 2^10

-   `--exploration_timesteps=` 2^3, 2^4, 2^6, 2^8, 2^10

![Random walk replay buffer[]{data-label="fig:buffers_random_walk"}](../../report/Study_1/buffer/sequential_16.png "fig:") | ![Random walk replay buffer[]{data-label="fig:buffers_random_walk"}](../../report/Study_1/buffer/sequential_256.png "fig:") | ![Random walk replay buffer[]{data-label="fig:buffers_random_walk"}](../../report/Study_1/buffer/sequential_4096.png "fig:")
|----------|-----------|------------|

Figure \[fig:buffers\_random\_walk\] shows replay buffer filled by
random walk exploration. And on the contrary to the uniform sampling
techniques, this one does not cover the entire space of the environment
when replay buffer are too small. But with big replay buffers, the two
techniques tend to give the same results. So by running DDPG on these
replay buffer we expect that small replay buffers give a very bad
results, but bigger the replay buffer will be much more the results
should approach the results of replay buffer uniformly sampled.

![Random walk exploration[]{data-label="fig:scores_random_walk"}](../../report/Study_1/1.2/curves1_2.png "fig:") | ![Random walk exploration[]{data-label="fig:scores_random_walk"}](../../report/Study_1/1.2/total_scores1_2.png "fig:")
|-----------|-------------|

On the contrary to the study with uniform sampling, here there is a
significant increase of the score according to the replay buffer size.
So it confirms our expectations, small replay buffer (here under 1024
replays) seems to be insufficient for DDPG to learn. And this is
probably because the agent has not enough timesteps to discover the
entire environment, it only knows the region it appeared so it cannot
know where rewards are. As seen in Figure \[fig:scores\_random\_walk\]
above, under the size of 256, experiences achieved very poor
performances moreover their respective standard deviation are far
superior to any other run with bigger replay buffer, which suggest that
barely no information could be learned from the environment with so
small replay buffer and random walk exploration. On the other hand, the
bigger replay buffer allowed DDPG to converge close to the optimal
policy as expected. An interesting note is that we can see no
significant difference between the 1024 long replay buffer and the sizes
above, this result might be explained by the simplicity of the
environment.
