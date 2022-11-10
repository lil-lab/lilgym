# *lil*Gym: Natural Language Visual Reasoning with Reinforcement Learning

[arXiv](https://arxiv.org/abs/2211.01994) | [code & data](https://github.com/lil-lab/lilgym) | [website](https://lil-lab.github.io/lilgym/)

baselines: coming soon

## Table of Contents

- [About](#about)
- [Examples](#examples)
- [Data](#data)
- [Codebase](#codebase)
- [Citation](#citation)

## About
We present *lil*Gym, a new benchmark for language-conditioned reinforcement learning in visual environments. *lil*Gym is based on 2,661 highly-compositional human-written natural language statements grounded in an interactive visual environment. We annotate all statements with executable Python programs representing their meaning to enable exact reward computation in every possible world state. 

Each statement is paired with multiple start states and reward functions to form thousands of distinct Markov Decision Processes of varying difficulty. 
 
We experiment with *lil*Gym with different models and learning regimes. Our results and analysis show that while existing methods are able to achieve non-trivial performance, *lil*Gym forms a challenging open problem. 

### Examples
TowerScratch (left), TowerFlipIt (right)

<img src="/media/images/lilgym_gold_tower_scratch_ex.gif" alt="tower-scratch" width="300"/> <img src="/media/images/lilgym_gold_tower_flipit_ex.gif" alt="tower-flipit" width="300"/>

ScatterScratch (left), ScatterFlipIt (right)

<img src="/media/images/lilgym_gold_scatter_scratch_ex.gif" alt="scatter-scratch" width="300"/> <img src="/media/images/lilgym_gold_scatter_flipit_ex.gif" alt="scatter-flipit" width="300"/>


## Data

The data and details can be found in: `lilgym/data/`.

A description can be found in [lilGym: Natural Language Visual Reasoning with Reinforcement Learning](https://arxiv.org/abs/2211.01994). The data is based on the [Cornell Natural Language Visual Reasoning (NLVR) Corpus v1.0 (Suhr et al. 2017)](https://aclanthology.org/P17-2034/) corpus.

## Codebase

### Installation

1. Create a conda environment (Python >= 3.7)
```
conda create -n lilgym python=3.7
conda activate lilgym
```

2. Clone the repo: `git clone https://github.com/lil-lab/lilgym.git`

3. Install the dependencies
```
cd lilgym
pip install -r requirements.txt
```

To install the package from source:
```
cd lilgym
pip install .
```

### Example

The environments follow standard Gym API.

Following is a short demo script:

```python
import gym
from lilgym.data.utils import get_data

data = get_data('tower', 'scratch', 'train')
env = gym.make("TowerScratch-v0", data=data, stop_forcing=False)

env.seed(1)
observation = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
```

There are four configurations: `TowerScratch`, `TowerFlipIt`, `ScatterScratch` and `ScatterFlipIt`. 

An example for each is given below:

```python
data = get_data('tower', 'flipit', 'train')
env = gym.make("TowerFlipIt-v0", data=data, stop_forcing=False)

data = get_data('scatter', 'scratch', 'train')
env = gym.make("ScatterScratch-v0", data=data, stop_forcing=False)

data = get_data('scatter', 'flipit', 'train')
env = gym.make("ScatterFlipIt-v0", data=data, stop_forcing=False)
```

There are three data splits for each configuration (i.e. `TowerScratch`, `TowerFlipIt`, `ScatterScratch` or `ScatterFlipIt`): `train`, `dev`, and `test`.

The option `stop_forcing` specifies whether to use the training algorithm with stop forcing. Inference is always done without stop forcing.

The baselines with the training and inference code will also be soon released.

### License
MIT

## Citation
```
@misc{wu2022lilgym,
      title={lilGym: Natural Language Visual Reasoning with Reinforcement Learning}, 
      author={Wu, Anne and Brantley, Kianté and Kojima, Noriyuki and Artzi, Yoav},
      year={2022},
      eprint={2211.01994},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Ackowledegment
This research was supported by ARO W911NF21-1-0106, NSF under grant No. 1750499, a gift from Open Philanthropy, and NSF under grant No. 2127309 to the Computing Research Association for the CIFellows Project. 
Results presented in this paper were obtained using CloudBank, which is supported by the National Science Foundation under award No. 1925001. 
We thank Alane Suhr, Ge Gao, Justin Chiu, Woojeong Kim, Jack Morris, Jacob Sharf and the Cornell NLP Group for support, comments and helpful discussions.

## Contact
Anne Wu (<annewu@cs.cornell.edu>)
