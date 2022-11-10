import numpy as np

from lilgym.envs.vars import EPS
from lilgym.envs.utils_action import is_stop


class Reward:
    def __init__(self, logical_form: str, target_bool: bool = True):
        self.lf = logical_form
        self.target_bool = target_bool

    def __call__(self, action, prediction: bool):
        return self.get_reward(action, prediction, self.target_bool)

    def get_reward(self, action, prediction: bool, target: bool):
        reward = 0.0
        if is_stop(action):
            if prediction == target:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = -EPS
        return reward
