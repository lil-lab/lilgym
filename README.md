# *lil*Gym: Natural Language Visual Reasoning with Reinforcement Learning

[arXiv](https://arxiv.org/abs/2211.01994) | code & data (coming on Wednesday Nov. 9, 2022) | [website](https://lil-lab.github.io/lilgym/)

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

---

For more details, please refer to the paper "[lilGym: Natural Language Visual Reasoning with Reinforcement Learning](https://arxiv.org/abs/2211.01994)", Anne Wu, Kianté Brantley, Noriyuki Kojima, Yoav Artzi.

### Examples
Tower-Scratch (left), Tower-FlipIt (right)

<img src="/media/images/lilgym_gold_tower_scratch_ex.gif" alt="tower-scratch" width="300"/> <img src="/media/images/lilgym_gold_tower_flipit_ex.gif" alt="tower-flipit" width="300"/>

Scatter-Scratch (left), Scatter-FlipIt (right)

<img src="/media/images/lilgym_gold_scatter_scratch_ex.gif" alt="scatter-scratch" width="300"/> <img src="/media/images/lilgym_gold_scatter_flipit_ex.gif" alt="scatter-flipit" width="300"/>


## Data
Coming on November 9, 2022.


## Codebase
Coming on November 9, 2022.


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
