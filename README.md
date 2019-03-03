# RL_analysis

## Installation

```sh
git clone https://github.com/schott97l/RL_analysis.git
cd RL_analysis && git submodule update --init
cd RL-evaluation-environment/gym-multi-dimensional && pip install -e .
```

## Submodules

[custom gym environment](https://github.com/hroussille/RL-evaluation-environment)

[reinforcement learning implementations](https://github.com/schott97l/RL_implementations)

## Usage Examples
To run a random policy on the 2D environment :
```sh
python run_multidimensional.py
```

To learn aand run a DDPG policy on the 2D environment with half high and low reward, and save the model and the results:
```sh
python learn_multidimensional.py --policy_name DDPG --output results1 --save --high_reward_count half --low_reward_count half
python run_multidimensional.py --policy_name DDPG --policy_directory results1/models --high_reward_count half --low_reward_count half
```

To learn and run a DDPG policy on the 2D environment with one high and one low reward, and save the model and the results:
```sh
python learn_multidimensional.py --policy_name DDPG --output results2 --save --high_reward_count one --low_reward_count one
python run_multidimensional.py --policy_name DDPG --policy_directory results2/models --high_reward_count one --low_reward_count one
```

## Studies
We did some experiments to study the behavior of DDPG algorithm learning according to different parameters.
We study the average reward per step according to the number of dimensions in an environment with half small and half big reward, and also in an environment with only one small and one big reward. We also study the effect of the replay buffer size on the ability of learning with a replay buffer filled by random exploration.

### Study 1
Average reward per step gained by agents who learned with DDPG algorithm on the environments according to the number of dimensions.

#### Study 1.1
On an environments with half small rewards and half big reward.
The details of the experiment are available in [study_1.1.sh](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_1.1/study_1.1.sh)
![reward/step according to dimensions number](https://lh3.googleusercontent.com/IuhnrDTJpFqRgSwEEpvqcJ0OjNAOmRKf74HAU6bgqRTNsQlyExtfpBOWxsfhgewEtW4KgM5ZXGorPA)

#### Study 1.2
On environments with one small rewards and one big reward.
The details of the experiment are available in [study_1.2.sh](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_1.2/study_1.2.sh)
![reward/step according to dimensions number](https://lh3.googleusercontent.com/6RA4bFGruiqC0fdrqJW8i0jrsBkrW9RtNGl33_LsQ9Lf6B3Gova4oNEUVCUm1Gdsgydd85dvkui3jw)

## Study 2
Average reward per step gained by agents who learned with DDPG algorithm on the environments according to the replay buffer size.
The details of the experiment are available in [study_2.sh](https://github.com/schott97l/RL_analysis/blob/master/Studies/Study_2/study_2.sh)
![reward/step according to replay buffer size](https://lh3.googleusercontent.com/sm4Ng-SHe_RiyQZvN8TlS5EkxiwvlY2OqxLTQykvQ79OFHdaE3zyLw6sKTgSRJhIuvIoCm5klCQgPw )

## Study 4
Average reward per step gained by agents who learned with DDPG algorithm on the environments according to the target network update rate.
![reward/step according to the target network update rate](https://lh3.googleusercontent.com/VJdtv-U66-MbyfVqLA1EOoyoQha5bEB9C6mQ5FUWHPFVwJj67T_Oq-4IT-_dIYPe1dPGwCL6RPdc9A)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc1MDk3NTc1MV19
-->