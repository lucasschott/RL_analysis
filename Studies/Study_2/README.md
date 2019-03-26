#  Study 2 

## Influence of replay buffer size on average reward

In this study we analyze the influence of the replay buffer size on the average reward. The learning step parameter is set to 40k , with a replay buffer size ranging from 16 to 64k. The environment is 2 dimensional with rewards present on the top and right edge.
The algorithm ( DDPG ) is not allowed to push new experiences to the replay buffer. Making the exploration phase the only source of information on the environment. This is made possible by the use of the --no-new-exp parameter. For a full list of the parameters used for this study , please refer to study_2.sh

We expect to  see very poor performance on the lower replay buffer sizes , as the number of experiences that can be contained in the replay buffer is too low , leading to an almost certain auto correlation of the experiences when sampling from the replay buffer.

As the replay buffer size increases , the performance should rise and settle to a level dependant of the environment. In our 2D environment , the starting position greatly influences the outcome and therefore the average reward obtained. We forced the starting point to be uniformly generated around the center point of the environment in a radius of 0.1 in order to reduced the variance that would otherwise be induced.  



|  Performances |
|:-:|
| ![reward-step](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualisations/total_scores.png)

As we can see , the reward / step seems to vee converging quite early. With a replay buffer of size 256, 40000 learning steps are enough to converge close to the optimal policy. The sizes above barely show any improvement.  



|  Performances per replay buffer size |
|:-:|
|![lol](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_replay&#32;buffer&#32;size.png) |

As seen in the graph above , the 16 and 64 experiences replay buffer achieved very poor performances. Their respective variance is far superior to any other run , which suggest that barely no information could be learned from the environment.

On the other hand the bigger replay buffer allowed DDPG to converge close to the optimal policy as expected. An interesting note is that we can see no significative difference between the 256 long replay buffer and the sizes above, this result might be explained by the simplicity of the environment.

Below is the resulting policy of the 64k replay buffer learning in both contour and gradient field format.

The environment reward are placed on the top and right edges.  



Contour according to replay buffer size           |  Gradient according to replay buffer size
:-------------------------:|:-------------------------:
![contour](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/q_contour_time_loop.gif)   |  ![gradient](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_time_replay_buffer_loop.gif)
