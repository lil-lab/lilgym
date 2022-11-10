from dataclasses import dataclass
from typing import List, Dict
from PIL.Image import Image


@dataclass
class ContextState:
    """
    A sample in the dataset, composed of: 
    - a context (a sentence, the corresponding logical form, and a target boolean)
    - a state (an image, with its structured representation)

    Example of img_struct:
    - [[], [], []] is a structured representation of an empty image.
    - [[{"y_loc": 58, "x_loc": 41, "size": 20, "type": "square", "color": "Yellow"}], [], []] 
    is a structured representation of an image with a yellow square of medium size in the 
    first box (the left box), where the upper-left coordinate is at (41, 58)
    """

    sentence: str
    lf: str
    img_struct: List[List]
    target_bool: bool = True
    img: Image = None
