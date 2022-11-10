from typing import List
import itertools
import numpy as np

from gym.spaces import Discrete, MultiDiscrete, Tuple
from lilgym.envs.utils_action import (
    Stop,
    TowerAdd,
    TowerRemove,
    ScatterAdd,
    ScatterRemove,
    to_tower_action,
    to_scatter_action,
)


TOWER_MULTI = [3, 3, 3]
SCATTER_MULTI_COORD = [19, 5]
SCATTER_MULTI_ITEM = [3, 3, 3]
SCATTER_MULTI = [3] + SCATTER_MULTI_COORD + SCATTER_MULTI_ITEM


class TowerActionSpace:
    def __init__(self):
        self._action_space = Tuple(
            (
                Discrete(3),  # 3 action types: ADD, REMOVE, STOP
                Discrete(3),  # 3 choices for box: LEFT, MIDDLE, RIGHT
                Discrete(3),  # 3 choices for color: YELLOW, BLACK, BLUE
            )
        )
        self.actions_dim = TOWER_MULTI
        self.shape = len(self.actions_dim)

    def get_shape(self):
        return self.shape

    def get_actions_dim(self):
        return self.actions_dim

    def sample(self):
        return np.asarray(self._action_space.sample())


class ScatterActionSpace:
    def __init__(self):
        self._action_space = Tuple(
            (
                Discrete(3),  # 3 action types: ADD, REMOVE, STOP
                MultiDiscrete(SCATTER_MULTI_COORD),  # Number of choices for (x, y)
                MultiDiscrete(
                    SCATTER_MULTI_ITEM
                ),  # Number of choices for (shape, color, size)
            )
        )
        self.actions_dim = SCATTER_MULTI
        self.shape = len(self.actions_dim)

    def get_shape(self):
        return self.shape

    def get_actions_dim(self):
        return self.actions_dim

    def sample(self):
        t = self._action_space.sample()
        # Flatten the tuple
        action_type = np.array([t[0]])
        action_coord_item = np.array(list(itertools.chain.from_iterable(t[1:])))
        action = np.concatenate((action_type, action_coord_item), axis=0)
        return action
