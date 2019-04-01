# Study 2:

In this study, we analyse the influence of a filtered replay buffer on the performances of DDPG on a 2 dimensional environment. The filter imposes a circular unexplored region in the environment. The flag --no-new-exp is set for the whole study , this forbids DDPG to add any experiences to the replay buffer. Meaning that when the exploitation phase starts and until it ends, all the experiences present in the replay buffer are static , none will be added or removed, Also known as Batch Reinforcement Learning.

Several configurations are studied, where we act on the filter position and size while using different exploration schemes. For each of them the learning timesteps parameter is set to 40 000, on a replay buffer that can contain up to 4096 experiences.

The environment being 2 dimensional where each dimension $d \in [-1, 1]$, we can visualize the content of a replay buffer to see the explored regions with the different exploration schemes and filter positions.

![](Study_2/visualizations/uniform.png) | ![](Study_2/2.1/visualizations/uniform-center.png) | ![](Study_2/2.3/visualizations/uniform-corner.png)
:---------------:|:---------------:|:----------------:
![](Study_2/visualizations/sequential.png) | ![](Study_2/2.2/visualizations/sequential-center.png) | ![](Study_2/2.4/visualizations/sequential-corner.png)

The exploration mode should play a critical role in the convergence rate, as the random walk exploration will explore a smaller portion of the environment than the uniform one on average. This phenomenon is more apparent as fewer experiences are sampled. Intuitively, the size of the filter should also influence the performance as DDPG will be required to interpolate or extrapolate ( depending on the filter position ) in order to converge to the optimal policy.

The filter that we use is circular, with $\theta$ the center of the filter, $r$ its radius and $x$ a point in the n dimensional space the indicator function that we use to know if a point is in the filter is :

\begin{equation}
\mathbbm{1}_F(x) := 
    \begin{cases}
      1,\:if\:\sqrt(\sum\limits_{i=1}^n (\theta_i - x_i) ^ 2)\:<\:r\\
      0,\:otherwise.
      \end{cases}
\end{equation}
 
 Therefore, we do allow points that are lying on the filter boundary.
 
We will compare the resulting policies to the optimal policy show in figure \ref{fig:optimal_policy}. And evaluate how the learned policy deviates from it for several key size and positions of filter.

## Uniform exploration and centered filter

This study focuses on the behaviour of DDPG on a replay buffer with uniform exploration and a  center filter with a radius ranging from 0 ( no filter ) to 1.2

![](Study_2/2.1/visualizations/scores_filter_size.png) | ![](Study_2/2.1/visualizations/total_scores.png)
:---------------:|:---------------:

On Figure \ref{fig:center_curves_uniform} we cannot see any significant difference on the performance achieved according to the filter size. It is however important to present the resulting $Q(s, \pi(s))$ for some filter size value to understand how DDPG adapted to the presence of the filter. Please note that the $\pi(s)$ visualizations are not included as they were all optimal.

![](Study_2/2.1/visualizations/Q_contour_0_4.png) | ![](Study_2/2.1/visualizations/Q_contour_0_8.png) | ![](Study_2/2.1/visualizations/Q_contour_1_2.png)
:---------------:|:---------------:|:----------------:

DDPG with a uniform exploration is perfectly able to converge even when a significant proportion of the environment is left unexplored due to a centered filter.
The estimated Q values seem to increase proportionally to the filter size on but there is no evidence to conclude that a divergent behaviour took place during the learning process.

We can clearly see that the centered filter had an effect on the learned policy on Figure \ref{fig:center_contour_uniform} by comparing this result with figure \ref{fig:optimal_policy} but those effect did not impact the performance. Mostly because of the simplicity of the environment and the independence of each dimensions with respect to the others.

## Random walk exploration and centered filter

This study focuses on the behaviour of DDPG on a replay buffer with random walk exploration and a  center filter with a radius ranging from 0 ( no filter ) to 1.2

![](Study_2/2.2/visualizations/scores_filter_size.png) | ![](Study_2/2.2/visualizations/total_scores.png)
:---------------:|:---------------:

The result are very similar to the previous case where no significant performance drop can be observed on Figure \ref{fig:center_curves_random_walk}. You will find below the resulting $Q(s, \pi(s))$ for the same sample filter size value.

![](Study_2/2.2/visualizations/Q_contour_0_4.png) | ![](Study_2/2.2/visualizations/Q_contour_0_8.png) | ![](Study_2/2.2/visualizations/Q_contour_1_2.png)
:---------------:|:---------------:|:----------------:

Interestingly on Figure  \ref{fig:center_contour_random_walk} , the filter with 0.4 radius seems to have been better handled with random walk exploration giving a less impacted resulting policy than with uniform exploration.

## Uniform exploration and cornered filter

This study focuses on the behaviour of DDPG on a replay buffer with uniform exploration and a cornered filter with a radius ranging from 0 ( no filter ) to 2.2. As the filter is now in the corner covering the most rewarding spot in the environment one could expect a drop in performance proportional to the filter size since it will mask a significant proportion of high rewarding positions to DDPG.

![](Study_2/2.3/visualizations/scores_filter_size.png) | ![](Study_2/2.3/visualizations/total_scores.png)
:---------------:|:---------------:

We can see observe the expected performance drop on Figure \ref{fig:corner_curves_uniform} when the filter size exceeds 2.0 while the sizes below do not seem to affect the performance in any significant way. You will find below the resulting $Q(s, \pi(s))$ for each filter size value.

![](Study_2/2.3/visualizations/Q_contour_0_4.png) | ![](Study_2/2.3/visualizations/Q_contour_0_8.png) | ![](Study_2/2.3/visualizations/Q_contour_1_2.png)
:---------------:|:---------------:|:----------------:
![](Study_2/2.3/visualizations/Q_contour_1_6.png) | ![](Study_2/2.3/visualizations/Q_contour_2_0.png) | ![](Study_2/2.3/visualizations/Q_contour_2_2.png)

As we can see on Figure \ref{fig:corner_contour_uniform} , until the filter reaches a critical size of 2.0, masking all dimensions boundaries that would lead to a high reward the Q values estimation is inverted with respect to the optimal one with an average estimated reward over all the observation space close to the value of a low reward. However, the policy taken by the actor is managing respectable performances.

When the filter radius goes above the 2.0 threshold we do notify some divergent behaviour as the estimated Q values are now clearly out of the allowed range for the real reward values. The performance drop at this point is significant since the actor is only going on the bottom left corner where the only alternative is a low reward.

Here are the gradient field visualizations of the actor's decisions for the 2.0 and 2.2

![](Study_2/2.3/visualizations/Pi_arrow_2_0.png) | ![](Study_2/2.3/visualizations/Pi_arrow_2_2.png)
:---------------:|:---------------:

It now becomes clear wiht Figure \ref{samples_policies_uniform_corner} why the 2.0 and 2.2 radius filters have such different impact on the performance. The 2.2 filter radius with no high reward samples forced DDPG to act uniformly in order to reach the bottom left corner where nothing but a low reward is present. The 2.0 radius with action relatively close to the optimal policy managed a better average score even though the Q values estimations did diverge. 

## Random walk exploration and cornered filter

This final study focuses on the behaviour of DDPG on a replay buffer with random walk exploration and a cornered filter with a radius ranging from 0 ( no filter ) to 2.2.
As we have seen earlier, we did notice a subtle difference between a random walk exploration and a uniform one on a centered filter.

![](Study_2/2.4/visualizations/scores_filter_size.png) | ![](Study_2/2.4/visualizations/total_scores.png)
:---------------:|:---------------:

A performance drop is clearly present on Figure  \ref{fig:corner_curves_random_walk} , earlier than with uniform exploration as both the 2.0 and 2.2 sized filters induced a lower average reward per step.

![](Study_2/2.4/visualizations/Q_contour_0_4.png) | ![](Study_2/2.4/visualizations/Q_contour_0_8.png) | ![](Study_2/2.4/visualizations/Q_contour_1_2.png)
:---------------:|:---------------:|:----------------:
![](Study_2/2.4/visualizations/Q_contour_1_6.png) | ![](Study_2/2.4/visualizations/Q_contour_2_0.png) | ![](Study_2/2.4/visualizations/Q_contour_2_2.png)


 As we can see on Figure \ref{fig:corner_contour_random_walk} the policy is impacted earlier, around 1.2 a shift is noticeable. The sized above increase that effect and when the 2.0 radius is used we start noticing a divergent behaviour in the Q values estimations.
 
![](Study_2/2.4/visualizations/Pi_arrow_1_6.png) | ![](Study_2/2.4/visualizations/Pi_arrow_2_0.png) ![](Study_2/2.4/visualizations/Pi_arrow_2_2.png)
:---------------:|:---------------:|:----------------:
 
 On Figure \ref{fig:sample_policies_corner_sequential} the 1.6 radius filter visualizations show again a gap between the Actor and the Critic representation of the observation space. On the 2.0 and 2.2 filter size have their actions directed to the bottom left corner where only the low reward is present which explains the lower score. Even though the Q values did diverge, it is coherent with the contour visualization where we clearly see a gradient towards the low reward.
