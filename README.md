# RL-analysis

## Installation

```sh
git clone https://github.com/schott97l/RL_analysis.git
cd RL_analysis && git submodule update --init
cd RL-evaluation-environment/gym-multi-dimensional && pip install -e .
```

## Submodules

[custom gym environment](https://github.com/hroussille/RL-evaluation-environment)
[reinforcement learning implementations](https://github.com/hroussille/RL_implementations)

## Usage Examples

To learn a DDPG policy on the environment:
```sh
python learn_multidimensional.py [options]
```
To run a policy that has been learned on the environment:
```sh
python run_multidimensional.py [options]
```
