# Project : Evaluation environment for deep reinforcement learning algorithm analysis
    
## Authors:

Hector Roussille\
Sorbonne University\
Paris, France\
hector.roussille@etu.upmc.fr 

Lucas Schott\
Sorbonne University\
Paris, France\
lucas.schott@etu.upmc.fr

Olivier Sigaud\
Supervisor\
Sorbonne University\
Paris, France\
Olivier.Sigaud@upmc.fr
    
## Abstract:
We analyze the Deep Deterministic Policy Gradient algorithm (DDPG),
which is a deep reinforcement learning algorithm (RL) with continuous
action space control, on a basic evaluation environment to see the
influence of several parameters on the learning abilities. Our
evaluation environment is an N-dimensional hypercube with rewards on its
hyperfaces, the simplicity of the environment allows us to study each
parameter separately and to provide an interpretation of the results. We
study in particular the extrapolation ability when DDPG learns with
sparse or missing data from fixed dataset, and also the influence of the
number of dimensions on the learning process.

Introduction
============

As the tendency is to test Reinforcement Algorithms on increasingly
complex environments, under the supervision of Olivier Sigaud we decided
to take the opposite stance. In this work we use a very basic
environment and explore how simple variations in parameters such as the
exploration scheme or the number of dimensions influence the
performances of state of the art algorithms like DDPG while being free
of complex behaviours internal to the environment.

We first analyse the influence of the replay buffer size on two
different exploration schemes using batch reinforcement learning. The
first one is a uniform exploration of the environment, the second one is
a random walk letting the untrained agent take random steps in the
environment until enough experiences have been acquired. Building on
those results we got interested in the way a learned policy is disturbed
by an unexplored region imposed by a filter applied to the exploration
phase. Finally the dimensionality of the environment becomes the subject
of interest where we study the influence on the performance of different
size of environment in terms of number of dimensions.

Our first objective in this project is to verify if DDPG will fail learn
correctly in our environment with a batch learning on a fixed dataset
without exploration as described in [@fujimoto_off-policy_2018]. The
second one is to analyse the variations in Q values learning when a
certain proportion of the environment is purposely left unexplored while
in batch reinforcement learning. The third and last objective is to
understand how DDPG’s performances vary according to the dimensionality
of the environment, the initial position of the agent and dependant
versus independant actions.

Specifications
==============

Evaluation environment
----------------------

For this project we have to create a configurable environment in order
to run reinforcement learning algorithms on it. The environment has to
respect these specifications:

-   A $N$ dimensional hypercube.

-   With low and high rewards on its opposite faces on some dimensions,
    or no rewards on other dimensions.

-   The goal of the agent must be to maximize its reward.

The agent must also be configurable:

-   It has to be controlled with either discrete or continuous actions.

-   It must have either a speed limit on each dimensions independently
    or a maximum speed vector norm.

-   It must have a velocity mode in which the agent’s only information
    is its current position, the actions are vectors representing a
    given speed in the n dimensional environment. With $S$ the
    observation vector and $A$ the velocity vector in the n dimensional
    space defined as:
    
    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/1.PNG)  

    The new observation, $S'$ is computed with : $S' = S + A * p$ where
    $p$ is the power typically set to 0.1 in velocity mode. Since each
    dimension is bounded by $k$ we finally apply the following
    
    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/2.PNG)  

-   It must have an acceleration mode in which the agent controls its
    acceleration and has information about both its current position and
    velocity, each action is an acceleration in a given direction. In
    this mode, the observation vector $S$ holds both the position and
    velocity in each dimension while $A$ now represents accelerations:

    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/3.PNG)  

    The acceleration is first scaled using the $p$ parameter, usually
    set to 0.01 in acceleration mode and then added to the current
    velocity :

    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/4.PNG)  

    Friction $\mathbbm F(x)$ is then applied on the velocity vector with
    $f$ the friction parameter set to 0.001 :
    
    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/7.PNG)

    Clipping is applied to the velocity vector too with M the velocity
    boundary usually set to $K$:

    ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/5.PNG)  

    Finally the position is updated and clipped :

   ![alt text](https://raw.githubusercontent.com/schott97l/RL_analysis/master/imagesReadme/6.PNG)  

The environment has to be compatible with the OpenAI Gym API, so that it
is usable with all reinforcement learning algorithms that also respect
that API.

Algorithms
----------

We have to use an available implementation of the DDPG algorithm, and
apply some modifications to add the required functionalities for our
project:

-   Modify the implementation of the replay buffer to be able to
    visualize it.

-   Be able to choose the exploration mode before training, between a
    uniform exploration and random walk.

We also have to add some function to help visualization:

-   Add a function which returns the action of the actor network over a
    discretization of the environment space in order to visualize the
    current policy.

-   Add a function which returns the estimated Q values of the critic
    network over a discretization of the observation space and the
    current policy.

Visualizations
--------------

Several visualization specific to the environment must be included:

-   A contour visualization of the Q values estimation returned by the
    modified DDPG / TD3. Alongside with an animation of its evolution
    with respect to time.

-   A gradient field visualization of the current policy of a DDPG gent
    once again with an animation of its evolution according to time.

-   A visualization of the content of an experience replay buffer
    displaying all the experiences contained with a visual indication of
    their seniority.

Studies
-------

We had to perform some studies about DDPG in our environment:

-   We had to study the behaviour of DDPG and its learning abilities on
    our 2D environment according to different sizes of replay buffer
    with both exploration modes which are uniform sampling and random
    walk.

-   Study DDPG’s learning abilities on our 2D environment when zones of
    the environment are hidden from the agent.

-   Study how DDPG learns according to the number of dimensions of the
    environment in terms of both average reward and convergence rate.

Environment
===========

The hypercube environment we created is implemented in Python3 and is
compatible with the OpenAI gym API [@noauthor_toolkit_2019]. The
environment allows real time visualization for 1D and 2D hypercubes, the
1D being a segment \[fig:1d\_env\] and the 2D being a square
\[fig:2d\_env\]. The rendering is done with the OpenAI classic controls
rendering engine based on PyOpenGL. Both the 1D and 2D rendering are
presented below.

![1 dimensional environment, 2 rewards[]{data-label="fig:1d_env"}](report/env/visualizations/1d.png)

![2 dimensional environment with different reward configurations[]{data-label="fig:2d_env"}](report/env/visualizations/2d_2reward.png "fig:") | ![2 dimensional environment with different reward configurations[]{data-label="fig:2d_env"}](report/env/visualizations/2d_1reward.png "fig:")
|---------------|--------------|

Algorithm
=========

We used an implementation of the DDPG algorithm from Scott Fujimoto
[@fujimoto_pytorch_2019], while some modifications were applied to add
the required functionalities. The core of those algorithms remains as
described in the original paper [@lillicrap_continuous_2015] that
introduced it. DDPG is a model-free deep reinforcement learning
algorithm, which runs on continuous environments with continuous
controls. It explores the environment and saves all its experiences in a
replay buffer. DDPG learns from the content of the replay buffer.

A replay buffer [@mnih_playing_2013] is a queue of size $N$ containing
experiences. An experience is a tuple
$({s}_{t},{a}_{t},{r}_{t},{s}_{t+1})$ where ${s}_{t}$ is a state,
${a}_{t}$ is the action taken in state ${s}_{t}$, ${r}_{t}$ is the
reward gained by taking this action on this state, ${s}_{t+1}$ is the
state resulting from the action. When the replay buffer is full, the
oldest sample are removed and the newest are added.

DDPG is composed of two neural networks, the actor and the critic. The
actor is the one which decides the action to perform, according to its
observations. It learns by maximizing the output of the critic network.
The critic learns the action value by propagating the rewards associated
with experiences in the replay buffer.

The implementation is in Python3, using the PyTorch library. The
functionalities that we added are related to data extraction and
configurability. The first one being $Q(s, \Pi(s))$ allowing us to
extract the Critic Q values approximations of the current policy
$\Pi(s)$, $s$ being a discretization of the observation space. Since a
visualization of the Actor’s decision might be useful, we introduced a
second functionality computing $\Pi(s)$ to extract and visualize the
Actor’s decisions over the same discretization of the observation space.

User documentation
==================

The source code of the project is available on
<https://github.com/schott97l/RL_analysis>. The repository contains two
independents submodules,
<https://github.com/schott97l/RL_implementations> and\
<https://github.com/hroussille/RL-evaluation-environment>, and the
scripts of the studies.

To run this project, Python3.6 or newer is required with PyTorch, Gym,
Numpy, Matplotlib and PyOpenGL:

    pip install numpy matplotlib torch torchvision pyopengl gym 

You can download the project and install it by running the following
commands:

    git clone https://github.com/schott97l/RL_analysis.git
    cd RL_analysis && git submodule update --init
    pip install -e RL-evaluation-environment/gym-hypercube

 \
The command below is used to learn a policy on our hypercube
environment:

    python learn_hypercube.py [parameters]

parameters:

  ---------------------- ----------------------------------------------- ---------------------------------------------------------------------------------------------------------
  parameter| description | default
  |-------------|-------------|----------|
  `--algorithm=<str>` | name of the algorithm to use, “TD3” or “DDPG” | default=“DDPG”
  `--seed=<int>` | seed for the random exploration | default=random
  `--dimensions=<int>` | number of dimensions of the environment | default=2“
  `--eval_freq=<int>` | how often (time steps) we evaluate | default=2e3
  `--exploration_timesteps=<int>` | exploration duration | default=1e4
  `--exploration_mode=<str>` | ”random\_walk“ or ”uniform“ | default=”random\_walk“
  `--learning_timesteps=<int>` | learning duration | default=1e4  
  `--buffer_size=<int>` | size of the replay buffer | default=1e4  
  `--no_new_exp` | learn from a fixed batch |  
  `--expl_noise=<float>` | noise | default=0.1  
  `--batch_size=<int>` | learning batch | default=64  
  `--discount=<float>` | discount factor | default=0.99  
  `--actor_hl1=<int>` | actor first hidden layer size | default=40  
  `--actor_hl2=<int>` | actor second hidden layer size | default=30
  `--critic_hl1=<int>` | critic first hidden layer size | default=40
  `--critic_hl2=<int>` | critic second hidden layer size | default=30
  `--learning_rate=<float>` | learning rate of the networks | default=1e-4
  `--tau=<float>` | target network update rate | default=5e-3
  `--policy_noise=<float>` | noise added to target policy during critic update | default=0.2
  `--noise_clip=<float>` | range to clip target policy noise | default=0.5
  `--policy_freq=<int>` | frequency of delayed policy updates | default=2
  `--quiet` | to not print on standard output |
  `--acceleration` | set acceleration mode | default: velocity
  `--discrete` | set discrete actions | default: continuous
  `--speed_limit_mode` | set max velocity mode, ”vector\_norm“ or ”independent“ | default=”vector\_norm“
  `--replay_buffer_visu` | visualize 2D replay buffer |
  `--no_policy_visu` | to not plot visualizations |
  `--no_render` | to not render the environment |
  `--save` | save logs, models or visualizations |
  `--output=<str>` | output directory | default=”results“
  `--high_reward_value=<float>` | value of the high reward | default=1
  `--low_reward_value=<float>` | value of the low reward | default=0.1
  `--high_reward_count=<str>` | ”half“ or ”one“ | default=”half“
  `--low_reward_count=<str>` | ”half“ or ”one“ | default=”half“
  `--mode=<str>` | rewards positions ”deterministic“ or ”random“ | default=”deterministic"
  `--reset_radius=<float>` | radius from the center for the agent spawn | default=None
  `--filter` | to add a circle filter of the replay buffer |
  `--filter_radius=<float>` | radius of the filter | default=0.2
  `--filter_pos=<float>` | position of the filter on the diagonal of the environment | default=0
  ---------------------- ----------------------------------------------- ---------------------------------------------------------------------------------------------------------

 \
The command below is used to run a policy which has been learned with
the previous script.

    python run_hypercube.py [parameters]

parameters:

  ---------------------------- ----------------------------------------------- -------------------------------------------------------------------------------------------
  parameter | description | default
  |-------------|-------------|----------|
  `--algorithm=<str>` | name of the algorithm to use, “TD3” or “DDPG” | default=“DDPG”
  `--policy_directory=<str>` | | default=“results/models”
  `--dimensions=<int>` | number of dimensions of the environment | default=2“
  `--max_episodes=<int>` | | default=50
  `--max_timesteps=<str>` | | default=1e4
  `--buffer_size=<str>` | size of the replay buffer | default=5e3
  `--quiet` | to not print on standard output |
  `--acceleration` | to set acceleration mode | default: velocity
  `--discrete` | to set discrete actions | default: continuous
  `--no_render` | to not render the environment |
  `--high_reward_value=<float>` | value of the high reward | default=1
  `--low_reward_value=<float>` | value of the low reward | default=0.1
  `--high_reward_count=<str>` | ”half“ or ”one“ | default=”half“
  `--low_reward_count=<str>` | ”half“ or ”one“ | default=”half“
  `--mode=<str>` | rewards positions ”deterministic“ or ”random“ | default=”deterministic"
  `--reset_radius=<float>` | radius from the center for the agent spawn | default=None
  ---------------------------- ----------------------------------------------- -------------------------------------------------------------------------------------------

Studies
=======

Theoretical modeling
--------------------

Before analysing any experimental result on the 2D environment, we first
decided to calculate the theoretical optimal value and policy by
discretizing the environment in order to perform the value iteration
algorithm on it [@bellman_markovian_1957]. We discretized the 2D
environment in a 80x80 grid.

![Theoretical optimal policy obtained by policy iteration[]{data-label="fig:theoretical_policy"}](report/discrete/discrete_contour.png "fig:") | ![Theoretical optimal policy obtained by policy iteration[]{data-label="fig:theoretical_policy"}](report/discrete/discrete_arrow.png "fig:")
|---------------|----------------|

Figure \[fig:theoretical\_policy\] show us the visualization of the
optimal state value and the associated policy. We can see that the more
we are close to the right or top edge greater the state value is, and
the optimal policy involved is to go straight to right or top edge (the
closest) to get the maximum reward.

Experimental methodology
------------------------

All the presented results are the results of 8 experiments obtained by
running the training with exactly the same parameters. The curves and
the errors bars shown are the average and the standard deviation over
these 8 experiments. The parameters used for each study are specified
before, the non specified ones are set to their default value as
described in the user documentation.

Study 1: Size of the replay buffer
----------------------------------
[README](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_1/README.md)

Study 2: Filtered replay buffer
-------------------------------
[README](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_2/README.md)

Study 3: Number of dimensions of the environment
-----------------------------
[README](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_3/README.md)

Conclusion
==========

Results
-------

We found a strong correlation between the size of the experience replay
buffer and the learning ability of DDPG in case of batch learning on a
fixed dataset, the simplicity of our environment make it also very easy
to learn with a small replay buffer. But contrary to
[@fujimoto_off-policy_2018], we did not found that learning from a fixed
dataset of experiences uncorrelated with the used policy leads to
extrapolation errors. We just saw differences between data collection
methods, learning from fixed dataset of experiences gathered by uniform
exploration works much better than learning from experiences gathered by
random walk, but when the size of the replay buffer increases, the
differences between random walk and uniform sampling disappear.

When analysing the behaviour of DDPG on a filtered environment we did
notice that DDPG’s interpolations abilities are far superior to its
extrapolation abilities. The centered filter was handled much better
than the corner one. However, we did not notice the divergent behaviour
that we were expecting as described in [@achiam_towards_2019].
Additionaly, the exploration scheme seemed to have no impact whatsoever
on the resulting performances.

The results of the study on the number of dimensions in the environment
on the learning ability of DDPG taught us more about the environment
than about DDPG. The reason why DDPG is unable to learn on our high
dimensional environments is not due to DDPG itself bue to the structure
of the environment where the actions of the agent are not independent.
The norm of the velocity vector being bounded by 1 if too many
dimensions are used when taking an action the resulting actions on each
dimensions will be negligible because of the normalisation step,
therefore the agent won’t make any progress. With independant actions
this problem is no more and DDPG is able to learn faster and reach
higher reward / step performances since it is able to combine its speed
along several dimensions.

Over all the experiment we did on our environment, it seems that
regardless of the dimensionality, exploration scheme or other parameters
DDPG will always learn a linear action value function. This leads to the
agent learning a linear policy that usually takes it towards the highest
rewarding corner of the environment. This way might be the easiest to be
guaranteed a high reward in the end but it is certainly not the fastest,
especially not in high dimensional environments. The behaviour we
expected was to go straight to the closest high reward, focusing only on
the dimension that matter in order to maximise the reward / step
parameter.

Future work
-----------

Future work could be to try to solve the problem of the linear policy
learned by DDPG, by trying bigger neural network than the default ones
or other activation functions than ReLu. Maybe the linear policy learned
by DDPG is due to the fact that our environment provides only positives
rewards whatever direction the agent chooses, then the ReLus could works
like linear functions. So further experiments have to be conducted to
understand why DDPG learns a linear policy instead of the optimal policy
computed with the value iteration algorithm on the discretized
environment.

After resolving the problem of linearity, with plenty of room for
improvements and ideas of additional studies we are for instance looking
to use Hindsight Experience Replay [@andrychowicz_hindsight_2017] on all
the existing studies to analyze the possible performance improvements.
Study 2 starts addressing some of the issues related to divergence on Q
values approximation [@achiam_towards_2019] over a filtered space but
different point of interest are also considered in order to validate the
results exposed by the team from Deep Mind on the so called deadly Triad
[@van_hasselt_deep_2018].

Adding more complexity and configurability to the environment would also
allow for deeper studies. By introducing a dependence between the
different dimensions and more reward distribution patterns to further
investigate the abilities and limits of DDPG.

## References

[1] S. Fujimoto, D. Meger, and D. Precup, “Off-Policy Deep Reinforcement Learning without Exploration,”
arXiv:1812.02900 [cs, stat], Dec. 2018.

[2] “A toolkit for developing and comparing reinforcement learning algorithms.: openai/gym,” Apr. 2019. original-
date: 2016-04-27T14:59:16Z.

[3] S. Fujimoto, “PyTorch implementation of TD3 and DDPG for OpenAI gym tasks: sfujim/TD3,” Mar. 2019.
original-date: 2018-02-22T18:15:37Z.

[4] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control
with deep reinforcement learning,” arXiv:1509.02971 [cs, stat], Sept. 2015.

[5] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing Atari
with Deep Reinforcement Learning,” arXiv:1312.5602 [cs], Dec. 2013.

[6] R. Bellman, “A Markovian Decision Process,” IUMJAB, vol. 6, no. 4, pp. 679–684, 1957.

[7] T. Schaul, D. Horgan, K. Gregor, and D. Silver, “Universal Value Function Approximators,” p. 9, July 2015.
21M AY 19, 2019

[8] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,”
arXiv:1802.09477 [cs, stat], Feb. 2018.

[9] J. Achiam, E. Knight, and P. Abbeel, “Towards Characterizing Divergence in Deep Q-Learning,” arXiv:1903.08894
[cs], Mar. 2019.

[10] M. Andrychowicz, D. Crow, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, O. P. Abbeel, and
W. Zaremba, “Hindsight Experience Replay,” p. 11, 2017.

[11] H. van Hasselt, Y. Doron, F. Strub, M. Hessel, N. Sonnerat, and J. Modayil, “Deep Reinforcement Learning and
the Deadly Triad,” arXiv:1812.02648 [cs], Dec. 2018.
