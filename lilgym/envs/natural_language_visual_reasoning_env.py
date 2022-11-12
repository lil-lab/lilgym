import time, copy
import numpy as np

import skimage.measure

import gym
from gym import spaces
from gym.utils import seeding

from lilgym.data.utils import get_data
from lilgym.envs.reward import Reward
from lilgym.envs.utils import (
    is_action_valid,
    get_action_space,
    is_terminal,
    compute_prediction,
    bool_from_string,
    can_force_stop,
)
from lilgym.envs.utils_image import get_base_image, draw_on_img
from lilgym.envs.vars import MAX_TIME_STEPS
from lilgym.envs.utils_state import ContextState
from lilgym.envs.utils_action import (
    to_tower_action,
    to_scatter_action,
    Action,
    TowerStop,
    ScatterStop,
    to_action_class,
    pad_action,
)


class NaturalLanguageVisualReasoningEnv(gym.Env):
    """
    An RL environment for Natural Language Visual Reasoning.
    """

    def __init__(
        self,
        env_opt: str,
        learn_opt: str,
        stop_forcing: bool,
        split: str = None,
        data: dict = None,
        evaluate: bool = False,
        horizon: int = MAX_TIME_STEPS,
    ):
        """
        Args:
            env_opt: Environment appearance option: "tower" or "scatter"
            learn_opt: The starting condition: "scratch" or "flipit"

            data: Dictionary of the context and initial states
            split: The data split ("train", "dev", or "test").
            Note: split and data are mutually exclusive, and split is considered in priority.

            stop_forcing: Whether stop forcing (SF) is used or not
            evaluate: Whether evaluation mode is on
        """
        print(f"{env_opt}-{learn_opt}-SF{stop_forcing} Environment initialized")

        self._env_opt = env_opt
        self._learn_opt = learn_opt
        self._stop_forcing = stop_forcing
        self._horizon = horizon

        self.action_space = get_action_space(self._env_opt)

        self._resize_width = 190
        self._resize_height = 50

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(self._resize_height, self._resize_width, 3)
                ),
                "sentence": spaces.Discrete(320),
                "target": spaces.Discrete(2),
            }
        )

        self._samples = {}
        assert (
            data or split
        ), "No data error: Either the split of the data needs to be specified, or the data needs to be given"

        if split:
            data = get_data(self._env_opt, self._learn_opt, split)
        assert data is not None, "Must provide environment initial states."

        for k in data.keys():
            if self._learn_opt == "scratch":
                self._samples[k] = ContextState(
                    data[k]["sentence"], data[k]["lf"], [[], [], []], True
                )
            elif self._learn_opt == "flipit":
                # The target bool will be converted to the inverse
                self._samples[k] = ContextState(
                    data[k]["sentence"],
                    data[k]["lf"],
                    data[k]["structured_rep"],
                    not bool_from_string(data[k]["label"]),
                )

        self._evaluate = evaluate
        self._evaluate_list = list(self._samples.keys())

        self._time_step = 0

        self._state = None

    def step(self, action):
        """
        Takes a step with the given action and returns next observation.
        """
        # If action is an iterable (ex. np.array or torch.Tensor), convert to an Type[Action] object
        if not isinstance(action, Action):
            action = pad_action(action, self._env_opt)
            action = to_action_class(action)

        # increment time step
        self._time_step += 1

        # compute reward
        prediction = False
        if self._state.img_struct:  # if the image is not empty
            prediction = compute_prediction(self._state.img_struct, self._state.lf)
        step_reward = self._reward_function(action, prediction)

        # Stop forcing
        force_stop = False
        if self._stop_forcing:
            force_stop = can_force_stop(
                action, prediction, target_bool=self._state.target_bool
            )
            if force_stop:
                if self._env_opt == "tower":
                    action = TowerStop()
                elif self._env_opt == "scatter":
                    action = ScatterStop()
                step_reward = self._reward_function(action, prediction)

        # Check if an action is invalid
        if not is_action_valid(self._env_opt, self._state.img_struct, action):
            force_stop = True
            step_reward = -1.0

        done = is_terminal(action, self._time_step, self._horizon, force_stop)

        if not done:
            # Image drawing / img_struct updating
            self._state = action.apply(self._state)

        info = {
            "sentence": self._state.sentence,
            "target": self._state.target_bool,
            "force_stop": force_stop,
        }
        if done:
            info["accuracy"] = (step_reward > 0.0) * 1.0
            info["accuracy_nosf"] = ((step_reward > 0.0) and not force_stop) * 1.0

        if self._evaluate:
            info["nb_to_evaluate"] = len(self._evaluate_list)

        return self._get_dict_obs(copy.deepcopy(self._state)), step_reward, done, info

    def _get_dict_obs(self, _state: ContextState):
        img = np.array(_state.img)

        image = np.zeros((50, self._resize_width, 3))
        image[:, :, :] = skimage.measure.block_reduce(img, (2, 2, 1), np.mean)

        return {
            "sentence": _state.sentence,
            "image": image,
            "target": 1 if self._learn_opt == "scratch" else int(_state.target_bool),
        }

    def _combine_sent_n_img(self, _state: ContextState):
        img = np.array(_state.img)

        state = np.zeros((51, self._resize_width, 3))
        state[:50, :, :] = skimage.measure.block_reduce(img, (2, 2, 1), np.mean)

        sent_embedding = np.zeros((self._resize_width, 3))
        sent_embedding[:, 0] = _state.embedding

        state[-1] = sent_embedding
        return state

    def reset(self, example_number=None):
        if example_number:
            return self.reset_example(example_number)

        if self._evaluate:
            item = self._evaluate_list.pop()
            self._state = self.reset_example(item)
        else:
            self._state = self.reset_example(
                list(self._samples.keys())[np.random.choice(len(self._samples))]
            )
        return self._get_dict_obs(self._state)

    def reset_example(self, example_number):
        """
        Resets the environment and starts a new episode.
        """
        self._time_step = 0
        self._state = copy.deepcopy(self._samples[str(example_number)])

        self._reward_function = Reward(
            self._state.lf, target_bool=self._state.target_bool
        )

        img, draw = get_base_image()

        if "flipit" in self._learn_opt:
            img = draw_on_img(img, self._state.img_struct)

        self._state.img = img
        return self._state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def close(self):
        pass
