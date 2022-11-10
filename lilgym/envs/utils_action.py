from abc import ABC, abstractmethod
import torch

from lilgym.envs.utils_image import (
    draw_item_tower,
    delete_item_tower,
    draw_item_scatter,
    delete_item_scatter,
)
from lilgym.envs.utils_state import ContextState


class Action(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(state: ContextState) -> ContextState:
        pass


class Stop(Action):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Stop"

    def apply(self, state: ContextState) -> ContextState:
        return state


class TowerAdd(Action):
    def __init__(self, box, color):
        self._box = box
        self._color = color

    def color(self):
        return self._color

    def box(self):
        return self._box

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = draw_item_tower(self, state.img, state.img_struct)
        return state

    def __repr__(self) -> str:
        return f"TowerAdd({self._box}, {self._color})"


class TowerRemove(Action):
    def __init__(self, box):
        self._box = box

    def box(self):
        return self._box

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = delete_item_tower(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f"TowerRemove({self._box})"


class ScatterAdd(Action):
    def __init__(self, x, y, shape, color, size):
        self._x = x
        self._y = y
        self._shape = shape
        self._color = color
        self._size = size

    def color(self):
        return self._color

    def x(self):
        return self._x

    def y(self):
        return self._y

    def shape(self):
        return self._shape

    def size(self):
        return self._size

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = draw_item_scatter(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f"ScatterAdd({self._x}, {self._y}, {self._shape}, {self._color}, {self._size})"


class ScatterRemove(Action):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = delete_item_scatter(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f"ScatterRemove({self._x}, {self._y})"


def is_stop(action):
    return type(action) == Stop


def is_add(action):
    return type(action) == TowerAdd or type(action) == ScatterAdd


def is_remove(action):
    return type(action) == TowerRemove or type(action) == ScatterRemove


def to_tower_action(raw_action):
    """
    For the Tower configuration.
    Takes a in_action (iterable) and convert to a Type[Action] object
    (Stop, TowerAdd, or TowerRemove).
    """
    if isinstance(raw_action, torch.Tensor):
        action = raw_action.cpu().detach().numpy()
    action_type = raw_action[0]
    if action_type == 0:
        return Stop()
    elif action_type == 1:
        box, color = raw_action[1], raw_action[2]
        return TowerAdd(box, color)
    elif action_type == 2:
        box = raw_action[1]
        return TowerRemove(box)
    else:
        raise ValueError(f"Invalid action type: {action_type}")


def to_scatter_action(raw_action):
    """
    For the Tower configuration.
    Takes a in_action (iterable) and convert to a Type[Action] object
    (Stop, TowerAdd, or TowerRemove).
    """
    if isinstance(raw_action, torch.Tensor):
        action = raw_action.cpu().detach().numpy()
    else:
        action = raw_action
    action_type = raw_action[0]
    if action_type == 0:
        return Stop()
    x, y = raw_action[1:3]
    if action_type == 1:
        shape, color, size = raw_action[3:]
        return ScatterAdd(x, y, shape, color, size)
    elif action_type == 2:
        return ScatterRemove(x, y)
    else:
        raise ValueError(f"Invalid action type: {action_type}")


def get_action_object(env_opt, in_action):
    """
    For both Tower and Scatter.
    Takes a in_action (iterable) and convert to a Type[Action] object.
    """
    if env_opt == "tower":
        return to_tower_action(in_action)
    elif env_opt == "scatter":
        return to_scatter_action(in_action)
