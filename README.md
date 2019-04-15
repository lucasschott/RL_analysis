# RL_analysis

## Installation

```sh
git clone https://github.com/schott97l/RL_analysis.git
cd RL_analysis && git submodule update --init
cd RL-evaluation-environment/gym-hypercube && pip install -e .
```

## Submodules

[custom gym environment](https://github.com/hroussille/RL-evaluation-environment)

[reinforcement learning implementations](https://github.com/schott97l/RL_implementations)

For more informations about those components , please refer to their respective repositories.

## Usage Examples
To run a random policy on the 2D environment :
```sh
python run_hypercube.py
```

To learn and run a DDPG policy on the 2D environment with half high and low reward, and save the model and the results:

```sh
python learn_hypercube.py --algorithm DDPG --save
python run_hypercube.py --algorithm DDPG
```

To learn and run a DDPG policy on the 2D environment with one high and one low reward, and save the model and the results:
```sh
python learn_hypercube.py --algorithm DDPG --output results_2 --save 
python run_hypercube.py --algorithm DDPG --policy_directory results_2/models
```

## Studies
We did some experiments to study the behavior of DDPG algorithm learning according to different parameters.
* Average reward / step according to the number of dimensions ( halft high reward / half low reward )
* Average reward / step according to the number of dimensions ( one high reward / one low reward )
* Average reward / step according to the replay buffer size
* Evolution of a policy on a filtered replay buffer


### Study 1 : Average reward / step according to the number of dimensions 
Average reward per step gained by agents who learned with DDPG algorithm on the environments according to the number of dimensions.

#### Study 1.1 ( halft high reward / half low reward )
On an environments with half small rewards and half big reward.
The details of the experiment are available in [study_1.1](https://github.com/schott97l/RL_analysis/tree/master/Studies/Study_1.1)

![reward/step according to dimensions number](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.1/visualizations/scores_dimensions.png)

#### Study 1.2 ( one high reward / one low reward )

On environments with one small rewards and one big reward.
The details of the experiment are available in [study_1.2](https://github.com/schott97l/RL_analysis/tree/master/Studies/Study_1.2)

![reward/step according to dimensions number](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_1.2/visualizations/scores_dimensions.png)

## Study 2 : Average reward / step according to the replay buffer size

Average reward per step gained by agents who learned with DDPG algorithm on the environments according to the replay buffer size.
The details of the experiment are available in [study_2](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_2)

![reward/step according to replay buffer size](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_2/visualizations/scores_replay%20buffer%20size.png)

## Study 3 : Evolution of a policy on a filtered replay buffer

Visualization of the learned policy by DDPG on a filtered replay buffer.

The replay buffer was filtered to impose an unexplored region in the center and in a corner.

The details of the experiment are available in [study_3](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_3)

### Replay buffer
Non filtered replay buffer           |  Filtered replay buffer
:-------------------------:|:-------------------------:
![non-filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/non-filtered.png)   |  ![filtered](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/images/filtered.png)

A second filter was used to filter the upper right portion of the environment. This corner is the meeting point of the two edges that will give a positive reward to the agent.

#### Sample Policy with a center filter
|  Contour | Gradient  |
|:-:|:-:|
| ![contour-lr-center](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/visualizations/lr-center-samples/samples/1/visualizations/Q_contour_time.gif) | ![policy-lr-center](https://raw.githubusercontent.com/schott97l/RL_analysis/master/Studies/Study_3/visualizations/lr-center-samples/samples/1/visualizations/Pi_arrow_time.gif) |

