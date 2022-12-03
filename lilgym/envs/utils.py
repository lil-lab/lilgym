from typing import List

from lilgym.envs.utils_image import SEP_WIDTH, BOX_SIZE
from lilgym.envs.utils_image import (
    can_draw_item_scatter,
    can_delete_item_scatter,
    get_box,
    convert_action_to_img_coordinates,
)

from lilgym.envs.logical_forms import *
from lilgym.envs.structured_rep import Image as NLVRImage
from lilgym.envs.structured_rep import ALMOST_TOUCHING_MARGIN
from lilgym.envs.structured_rep_enums import (
    Size,
    Color,
    Shape,
    Location,
    Relation,
    Side,
)

from lilgym.envs.action_space import (
    TowerActionSpace,
    ScatterActionSpace,
    TOWER_MULTI,
    SCATTER_MULTI,
)
from lilgym.envs.utils_action import is_stop, is_add, is_remove


def bool_from_string(tf_str):
    return tf_str == "true"


def compute_prediction(img_struct: List, expression: str):
    """
    Based on the code of Weakly Supervised Semantic Parsing with Abstract Examples, Goldman et al., 2019.

    :param expression: a logical form (string)
    :param image: an object of type Image (a strutured representation of an image)
    :return: the result of executing the logical form on the structured representation
    """
    result = False
    # create constants
    img_struct = NLVRImage(img_struct)

    all_boxes = img_struct.get_all_boxes()
    all_items = img_struct.get_all_items()
    result = eval(
        expression, globals().update({"all_boxes": all_boxes, "all_items": all_items})
    )

    if type(result) is not bool:
        raise TypeError("parsing returned a non boolean type")
    return result


def get_action_space(appearance: str, seed: int = 1):
    if appearance == "tower":
        return TowerActionSpace(seed=seed)
    elif appearance == "scatter":
        return ScatterActionSpace(seed=seed)


def is_terminal(action, force_stop: bool = False):
    """
    Check whether the rollout is reaching an end state, i.e. if:
    - the action taken is Stop
    - or we want to do stop forcing
    """
    return is_stop(action) or force_stop


def is_truncated(time_step: int, horizon: int):
    """
    Check whether the rollout is reaching a truncation condition i.e. if:
    - the horizon is reached
    """
    return time_step == horizon


def can_force_stop(last_action, prediction: bool, target_bool: bool = True):
    """
    Check whether we could do stop forcing, i.e. if:
    a goal state is reached AND the action taken is not Stop
    """
    return not is_stop(last_action) and (prediction == target_bool)


def is_action_valid(appearance: str, img_struct: List, action):
    """
    Check whether an action is valid.
    Returns:
        True for valid, False for invalid, -1 for irrelevant.
    """
    if img_struct == None:
        return -1

    # Special case: stop action is always valid
    if is_stop(action):
        return True

    if appearance == "tower":
        box = action.box()
        # Case 1: remove when there's nothing
        if is_remove(action) and len(img_struct[box]) == 0:
            return False

        # Case 2: add in a box when there's already 4 items
        if is_add(action) and len(img_struct[box]) == 4:
            return False
        return True
    elif appearance == "scatter":
        # The case where the action is on the separator
        box = get_box(action.x())
        # x is on the separator
        if box == -1:
            return 0

        # Case 1: remove when there's no element to remove at that place
        x, y = action.x(), action.y()
        if is_remove(action):
            x, y = action.x(), action.y()
            img_x, img_y = convert_action_to_img_coordinates(x, y, box)
            return any(
                el["x_loc"] == img_x and el["y_loc"] == img_y for el in img_struct[box]
            )

        # Case 2: Add an object when it's not possible
        # i.e.: add a 30x30px (large) object in a 20x20px cell, at the right/bottom of a box
        if is_add(action):
            size = action.size()
            img_x, img_y = convert_action_to_img_coordinates(x, y, box)
            if size == 2 and (img_x == 80 or img_y == 80):
                return False

            # Case 3: when it's impossible to add a shape (e.g. on top of another)
            shape, color = action.shape(), action.color()
            if not can_draw_item_scatter(action, None, img_struct):
                return False
        elif is_remove(action):
            if not can_delete_item_scatter(action, None, img_struct):
                return False
        return True
