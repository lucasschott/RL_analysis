# RL-analysis

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

### Study 1
Analyze average reward per step gained by agents who learned with DDPG algorithm on the environments according to the number of dimensions.

#### Study 1.1
On environments with half small rewards and half big reward.
The details of the experiment are described in study_1.1.sh.
![reward/step according to dimensions number](https://lh3.googleusercontent.com/IuhnrDTJpFqRgSwEEpvqcJ0OjNAOmRKf74HAU6bgqRTNsQlyExtfpBOWxsfhgewEtW4KgM5ZXGorPA)

#### Study 1.2
On environments with one small rewards and one big reward.
The details of the experiment are described in study_1.2.sh.

## Study 2
Analyze average reward per step gained by agents who learned with DDPG algorithm on the environments according to the replay buffer size.
The details of the experiment are described in study_2.sh.
![reward/step according to replay buffer size](https://lh3.googleusercontent.com/sm4Ng-SHe_RiyQZvN8TlS5EkxiwvlY2OqxLTQykvQ79OFHdaE3zyLw6sKTgSRJhIuvIoCm5klCQgPw )
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4ODkzMzg2NDQsLTEzNzA3OTQzNzIsMj
AxMTk3MjA0NF19
-->