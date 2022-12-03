# Environment

## Table of Contents

- [Basic usage](basic-usage)

## Basic usage

### Configurations

There are four configurations: `TowerScratch`, `TowerFlipIt`, `ScatterScratch` and `ScatterFlipIt`. An example for each:

```python
env = gym.make("TowerFlipIt-v0", split="train", stop_forcing=False)

env = gym.make("ScatterScratch-v0", split="dev", stop_forcing=False)

env = gym.make("ScatterFlipIt-v0", split="test", stop_forcing=False)
```

### Data splits

There are three data splits for each configuration: `train`, `dev`, and `test`.

### Arguments

**Stop forcing**

`stop_forcing` specifies whether to use the algorithm with stop forcing at training time. Inference is always done without stop forcing.

**Data reading**

There are two ways to load data:

1. Using the argument `split`

2. Using the argument `data`, with an example:

```python
import gymnasium as gym
from lilgym.data.utils import get_data

data = get_data('tower', 'scratch', 'train')
env = gym.make("TowerScratch-v0", data=data, stop_forcing=True, disable_env_checker=True)
```

### Action representations

There are 2 representations for the actions: as an object of type `Type[Action]` (easier to read), or as an iterable (numpy array).

For instance, the action `ADD(LEFT, BLACK)` can be represented as:
- `TowerAdd("LEFT", "BLACK")`
- or an array: `[1, 0, 1]`.

For both the Tower and Scatter configurations, we have:
- `STOP` actions: `TowerStop`, `ScatterStop`
- `ADD` actions: `TowerAdd`, `ScatterAdd`
- `REMOVE` actions: `TowerRemove`, `ScatterRemove`

**Syntax example**

```python
# Tower
>>> TowerStop()
>>> TowerAdd("LEFT", "BLACK") # Or TowerAdd(0, 1) also works
>>> TowerRemove("RIGHT") # Or TowerRemove(2)

# Scatter
>>> ScatterStop()
# The 1st argument in ScatterAdd is the x-coordinate, and 2nd is the y-coordinate.
>>> ScatterAdd(0, 0, "CIRCLE", "YELLOW", "SMALL") # Or ScatterAdd(0, 0, 0, 0, 0)
>>> ScatterRemove(0, 1)
```

**Example of a Scatter grid**

- RGB image: 380x100px
- Grid: 19x5
- Size of each cell: 20x20px

![](../../media/images/initial_state_grid.png)

#### Action type conversions

1. To convert from `Type[Action]` to `np.array`:

```python
>>> action = TowerAdd("LEFT", "BLACK") # It can also be TowerAdd(0, 1)

>>> action_array_rep = action.to_array()
>>> action_array_rep
array([1, 0, 1])
```

2. To convert from an action with an iterable representation (numpy array or Torch Tensor) to `Type[Action]`:

```python
from numpy import array
from lilgym.envs.utils_action import to_action_class

>>> action_array_rep = array([1, 0, 1])

>>> action = to_action_class(action_array_rep)
>>> action
TowerAdd("LEFT", BLACK")
```

#### 
