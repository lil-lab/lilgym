from abc import ABC, abstractmethod
import torch
import numpy as np

from lilgym.envs.utils_image import (
    draw_item_tower,
    delete_item_tower,
    draw_item_scatter,
    delete_item_scatter,
)
from lilgym.envs.utils_state import ContextState
from lilgym.envs.structured_rep_enums import Color, Size, Shape, TowerBox


class Action(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(state: ContextState) -> ContextState:
        pass


class TowerStop(Action):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "TowerStop()"

    def to_array(self):
        # The action STOP is represented by 0, and the -1 are
        # placeholder values so that the array length is
        # the same as for TowerAdd and TowerRemove
        return np.array([0, -1, -1])

    def apply(self, state: ContextState) -> ContextState:
        return state


class ScatterStop(Action):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "ScatterStop()"

    def to_array(self):
        # The action STOP is represented by 0, and the -1 are
        # placeholder values so that the array length is
        # the same as for ScatterAdd and ScatterRemove
        return np.array([0, -1, -1, -1, -1, -1])

    def apply(self, state: ContextState) -> ContextState:
        return state


class TowerAdd(Action):
    def __init__(self, box, color):
        """
        The arguments can either be int or str (for ease of use).

        Args:
            str:
                box: "LEFT", "MIDDLE", or "RIGHT"
                color: "YELLOW", "BLACK" or "BLUE"
            
            int:
                box: 0 (LEFT), 1 (MIDDLE), 2 (RIGHT)
                color: 0 (YELLOW), 1 (BLACK), 2 (BLUE)
        """
        if type(box) == str and type(color) == str:
            box = TowerBox.str_to_int(box)
            color = Color.str_to_int(color)
        self._box = box
        self._color = color
        self._box_str = TowerBox.int_to_str(self._box)
        self._color_str = Color.int_to_str(self._color)

    def color(self):
        return self._color

    def box(self):
        return self._box

    def to_array(self):
        # The action ADD is represented by 1
        return np.array([1, self._box, self._color])

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = draw_item_tower(self, state.img, state.img_struct)
        return state

    def __repr__(self) -> str:
        return f'TowerAdd("{self._box_str}", {self._color_str}")'


class TowerRemove(Action):
    def __init__(self, box):
        if type(box) == str:
            box = TowerBox.str_to_int(box)
        self._box = box
        self._box_str = TowerBox.int_to_str(self._box)

    def box(self):
        return self._box

    def to_array(self):
        # The action REMOVE is represented by 2, and the -1 are
        # placeholder values so that the array length is
        # the same as for TowerAdd
        return np.array([2, self._box, -1])

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = delete_item_tower(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f'TowerRemove("{self._box_str}")'


class ScatterAdd(Action):
    def __init__(self, x, y, shape, color, size):
        """
        The arguments can either be int or str (for ease of use).

        Args:
            str:
                x: x-coordinate of the item
                y: y-coordinate of the item
                shape: shape of the item, can be: "CIRCLE", "SQUARE", "TRIANGLE"
                color: "YELLOW", "BLACK", "BLUE"
                size: "SMALL", "MEDIUM", "LARGE"
            
            int:
                x
                y
                shape: 0 (CIRCLE), 1 (SQUARE), 2 (TRIANGLE)
                color: 0 (YELLOW), 1 (BLACK), 2 (BLUE)
                size: 0 (SMALL), 1 (MEDIUM), 2 (LARGE)
        
        Note: if a grid approximation is used, (x, y) corresponds to the upper-left coordinates 
        of the cell.
        Ex: Using a 19x5 grid to approximate the 380x100px RGB image:
            - (0, 0) is the cell at the top-left corner of the image
            - (1, 1) is the cell on the 2nd row and 2nd column (the upper-left coordinates on the 
            RGB image is (20, 20) in pixels)
        """
        if type(shape) == str and type(color) == str and type(size) == str:
            shape = Shape.str_to_int(shape)
            color = Color.str_to_int(color)
            size = Size.str_to_int(size)
        self._x = x
        self._y = y
        self._shape = shape
        self._color = color
        self._size = size
        self._shape_str = Shape.int_to_str(self._shape)
        self._color_str = Color.int_to_str(self._color)
        self._size_str = Size.int_to_str(self._size)

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

    def to_array(self):
        return np.array([1, self._x, self._y, self._shape, self._color, self._size])

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = draw_item_scatter(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f'ScatterAdd({self._x}, {self._y}, "{self._shape_str}", "{self._color_str}", "{self._size_str}")'


class ScatterRemove(Action):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def to_array(self):
        return np.array([2, self._x, self._y, -1, -1, -1])

    def apply(self, state: ContextState) -> ContextState:
        state.img_struct, state.img = delete_item_scatter(
            self, state.img, state.img_struct
        )
        return state

    def __repr__(self) -> str:
        return f"ScatterRemove({self._x}, {self._y})"


def is_stop(action):
    return type(action) == TowerStop or type(action) == ScatterStop


def is_add(action):
    return type(action) == TowerAdd or type(action) == ScatterAdd


def is_remove(action):
    return type(action) == TowerRemove or type(action) == ScatterRemove


def to_tower_action(raw_action):
    """
    For the Tower configuration.
    Takes a raw_action (iterable) and convert to a Type[Action] object
    (Stop, TowerAdd, or TowerRemove).
    """
    if isinstance(raw_action, torch.Tensor):
        action = raw_action.cpu().detach().numpy()
    action_type = raw_action[0]
    if action_type == 0:
        return TowerStop()
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
    Takes a raw_action (iterable) and convert to a Type[Action] object
    (Stop, TowerAdd, or TowerRemove).
    """
    if isinstance(raw_action, torch.Tensor):
        action = raw_action.cpu().detach().numpy()
    else:
        action = raw_action
    action_type = raw_action[0]
    if action_type == 0:
        return ScatterStop()
    x, y = raw_action[1:3]
    if action_type == 1:
        shape, color, size = raw_action[3:]
        return ScatterAdd(x, y, shape, color, size)
    elif action_type == 2:
        return ScatterRemove(x, y)
    else:
        raise ValueError(f"Invalid action type: {action_type}")


def pad_action(action, appearance):
    """
    For a Tower Action, pad to length 3 if it is not already, or
    for a Scatter Action, pad to length 6
    action: an iterable (np.array or Torch Tensor)
    """
    if appearance == "tower":  # pad to length 3
        if len(action) == 3:
            return action
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        return np.pad(action, (0, 3 - len(action)), constant_values=(-1,))
    elif appearance == "scatter":  # pad to length 6
        if len(action) == 6:
            return action
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        return np.pad(action, (0, 6 - len(action)), constant_values=(-1,))
    else:
        raise ValueError(f"Invalid appearance: {appearance}, should be 'tower' or 'scatter'")


def to_action_class(action):
    """
    For both Tower and Scatter.
    Takes an action (iterable) and convert to a Type[Action] object.
    """
    if len(action) == 3:  # Tower action
        return to_tower_action(action)
    elif len(action) == 6:  # Scatter action
        return to_scatter_action(action)
    else:
        raise ValueError(
            f"The action {action} should either be of length 3 (for Tower) or 6 (for Scatter)"
        )
