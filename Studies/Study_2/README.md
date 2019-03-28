#  Study 2 

## Influence of replay buffer size on average reward

In this study we analyze the influence of the replay buffer size on the average reward. The learning timestep parameter is set to 40k , with a replay buffer size ranging from 16 to 64k. The environment is 2 dimensional with rewards present on the top and right edge.
The algorithm ( DDPG ) is not allowed to push new experiences to the replay buffer. Making the exploration phase the only source of information on the environment. This is made possible by the use of the --no-new-exp parameter. For a full list of the parameters used for this study , please refer to study_2.sh

We expect to  see very poor performance on the lower replay buffer sizes , as the number of experiences that can be contained in the replay buffer is too low , leading to an almost certain auto correlation of the experiences when sampling from the replay buffer.

As the replay buffer size increases , the performance should rise and settle to a level dependant of the environment. In our 2D environment , the starting position greatly influences the outcome and therefore the average reward obtained. We forced the starting point to be uniformly generated around the center point of the environment in a radius of 0.1 in order to reduced the variance that would otherwise be induced.  



|  Performances |
|:-:|
| ![reward-step](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/total_scores.png)

As we can see , the reward / step seems to be converging quite early. With a replay buffer of size 256, 40000 learning steps are enough to converge close to the optimal policy. The sizes above barely show any improvement.  



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

For more in depth results , here are the separated learned policies and their respective learning scores :

### Buffer size : 16

#### Performance
![scores-16](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_16.png)

#### Average critic according to current policy

![contour-16](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_16.png)

#### Average policy
![gradient-16](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_16.png)


### Buffer size : 64
#### Performance
![scores-64](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_64.png)

#### Average critic according to current policy

![contour-64](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_64.png)

#### Average policy
![gradient-64](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_64.png)


### Buffer size : 256
#### Performance
![scores-256](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_256.png)

#### Average critic according to current policy


![contour-256](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_256.png)

#### Average policy
![gradient-256](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_256.png)


### Buffer size : 1024

#### Performance
![scores-1024](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_1024.png)

#### Average critic according to current policy


![contour-1024](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_1024.png)

#### Average policy
![gradient-1024](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_1024.png)


### Buffer size : 4096

#### Performance
![scores-4096](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_4096.png)

#### Average critic according to current policy


![contour-4096](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_4096.png)

#### Average policy
![gradient-4096](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_4096.png)


### Buffer size : 16384

#### Performance
![scores-16384](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_16384.png)

#### Average critic according to current policy


![contour-16384](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_16384.png)

#### Average policy
![gradient-16384](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_16384.png)


### Buffer size : 65536
#### Performance
![scores-65536](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_65536.png)

#### Average critic according to current policy


![contour-65536](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Q_contour_65536.png)

#### Average policy
![gradient-65536](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/Pi_arrow_65536.png)
