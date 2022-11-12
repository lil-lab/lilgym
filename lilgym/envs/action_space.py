from typing import List
import itertools
import numpy as np

from gym.spaces.space import Space

from lilgym.envs.action_spaces import TOWER_DEFAULT_ACTIONS, SCATTER_DEFAULT_ACTIONS


# Tower actions
# - 3 choices for action type (STOP, ADD, REMOVE)
# - 3 choices for box in which to exercise the action (LEFT, MIDDLE, RIGHT)
# - 3 choices for the item color (YELLOW, BLACK, BLUE)
TOWER_MULTI = [3, 3, 3]

# Scatter actions (Using the 19x5 grid simplification, with each cell being 20x20px)
# Coordinates-related arguments:
# - 19 choices for x on the grid, starting from the left (ex: 0 on the grid refer to
# the cells where the leftmost pixel is the leftmost pixel on the RGB image, then 1
# on the grid refer to the cells where the leftmost pixel is the 20th pixel on the
# RGB image starting from the left, etc.)
# - 5 choices for y on the grid, starting from the top (ex: 0 on the grid refer to
# the cells where the top pixel is the top pixel on the RGB image)
#
# Items-related arguments:
# - 3 choices for the item's shape (CIRCLE, SQUARE, TRIANGLE)
# - 3 choices for the item's color (YELLOW, BLACK, BLUE)
# - 3 choices for the item's size (SMALL, MEDIUM, LARGE)
SCATTER_MULTI_COORD = [19, 5]
SCATTER_MULTI_ITEM = [3, 3, 3]
SCATTER_MULTI = [3] + SCATTER_MULTI_COORD + SCATTER_MULTI_ITEM


class TowerActionSpace(Space):
    def __init__(self, seed=None):
        self._action_space = TOWER_DEFAULT_ACTIONS
        self.actions_dim = TOWER_MULTI
        super().__init__(shape=[len(self.actions_dim)], seed=seed)

    def get_shape(self):
        return self.shape

    def get_actions_dim(self):
        return self.actions_dim

    def sample(self):
        return self.np_random.choice(self._action_space)


class ScatterActionSpace(Space):
    def __init__(self, seed=None):
        self._action_space = SCATTER_DEFAULT_ACTIONS
        self.actions_dim = SCATTER_MULTI
        super().__init__(shape=[len(self.actions_dim)], seed=seed)

    def get_shape(self):
        return self.shape

    def get_actions_dim(self):
        return self.actions_dim

    def sample(self):
        return self.np_random.choice(self._action_space)
