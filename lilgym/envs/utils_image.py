from copy import deepcopy

from PIL import Image as PILImage
from PIL import ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from lilgym.envs.structured_rep import ALMOST_TOUCHING_MARGIN
from lilgym.envs.structured_rep_enums import Shape, Color, Size


NUM_BOXES = 3

# Ratio of box dimensions (assumed a square) to the smallest object size.
BOX_OBJECT_RATIO = 10

BOX_SIZE = 100
SEP_WIDTH = 40

OVERLAP_THRES = 0.5

# Size of 1 cell within the Scatter grid approximation
# 380px is the width of the original image, divided into 19 cells (number of choices for action x)
# 100px is the height of the original RGB image, divided into 5 cells (number of choices for action y)
CELL_SIZE = 20  # = 380 / 19 or 100 / 5


def draw_on_img(img, img_struct):
    """
    Draw the objects in the boxes on the PIL image.

    Args:
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
    
    Returns:
        img (PIL Image): modified image
    """
    boxes = []
    for box in img_struct:
        new_box = []
        for obj in box:
            new_box.append(obj)
        boxes.append(new_box)

    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x_offset = int(BOX_SIZE * i + SEP_WIDTH * i)
        for obj in box:
            x_start = x_offset + obj["x_loc"]
            y_start = obj["y_loc"]

            # -1 because the methods in draw all include the second coordinate
            x_end = int(x_start + obj["size"] / 10 * BOX_SIZE / BOX_OBJECT_RATIO) - 1
            y_end = int(y_start + obj["size"] / 10 * BOX_SIZE / BOX_OBJECT_RATIO) - 1

            if obj["type"] == Shape.CIRCLE.value:
                draw.ellipse(
                    [x_start, y_start, x_end, y_end],
                    fill=Color[Color(obj["color"]).name].as_rgb(),
                )
            elif obj["type"] == Shape.SQUARE.value:
                draw.rectangle(
                    [x_start, y_start, x_end, y_end],
                    fill=Color[Color(obj["color"]).name].as_rgb(),
                )
            elif obj["type"] == Shape.TRIANGLE.value:
                bottom_left = (x_start, y_end)
                bottom_right = (x_end, y_end)
                top = (x_end - int((x_end - x_start) / 2), y_start)
                draw.polygon(
                    [bottom_left, bottom_right, top],
                    fill=Color[Color(obj["color"]).name].as_rgb(),
                )
    return img


def draw_item_tower(action, img, img_struct):
    """
    Add an item according to a TowerAdd action to the PIL image,
    and to the structured representation of the image.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
    
    Returns:
        img_struct (Dict[Dict[Dict]]): img_struct with item drawn (added)
        img (PIL Image)
    """
    box = action.box()
    color = action.color()

    curr_size = 20
    curr_x_loc = 40
    curr_type = "square"
    curr_color = Color.int_to_color(color)

    # Find the box
    box_to_modify = img_struct[box]

    # Check if there's element and get the last element
    if len(box_to_modify) != 0:
        prev_y_loc = box_to_modify[-1]["y_loc"]
        if prev_y_loc == 17:  # Cannot add anymore
            return img_struct, draw_on_img(img, img_struct)
    else:
        prev_y_loc = 101  # 101-21 = 80

    curr_obj = {
        "x_loc": curr_x_loc,
        "y_loc": prev_y_loc - 21,
        "type": curr_type,
        "color": curr_color,
        "size": curr_size,
    }

    img_struct[box].append(curr_obj)
    return img_struct, draw_on_img(img, img_struct)


def delete_item_tower(action, img, img_struct):
    """
    Remove an item according to a TowerRemove action from the PIL image,
    and from the structured representation of the image.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
    
    Returns:
        img_struct (Dict[Dict[Dict]]): img_struct with item deleted
        img (PIL Image)
    """
    box = action.box()

    curr_size = 20
    curr_x_loc = 40
    curr_type = "square"
    curr_color = "Gray"

    # Find the box
    box_to_modify = img_struct[box]

    # Check if there's element / get the last element
    if len(box_to_modify) == 0:
        return img_struct, draw_on_img(img, img_struct)

    img_struct_for_delete = deepcopy(img_struct)

    # Get the position of the latest block
    prev_y_loc = box_to_modify[-1]["y_loc"]

    item_to_modify = {
        "size": curr_size,
        "x_loc": curr_x_loc,
        "type": curr_type,
        "y_loc": prev_y_loc,
        "color": curr_color,
    }

    img_struct[box].pop()
    img_struct_for_delete[box].append(item_to_modify)

    return img_struct, draw_on_img(img, img_struct_for_delete)


def get_base_image():
    """
    At reset time, get the base image with gray background and the 2 box delimiters.
    """
    img = PILImage.new(
        "RGB", (int(BOX_SIZE * NUM_BOXES + SEP_WIDTH * (NUM_BOXES - 1)), BOX_SIZE)
    )

    draw = ImageDraw.Draw(img)

    # Draw the base image: three boxes and two separating rectangles.
    for i in range(NUM_BOXES):
        x_start = int(BOX_SIZE * i + SEP_WIDTH * i)
        x_end = int(x_start + BOX_SIZE)
        # Width has to be set, otherwise PILImage will add 1 more pixel to the right of the
        # separator columns
        draw.rectangle([x_start, 0, x_end, BOX_SIZE], fill=(211, 211, 211, 255))

    for i in range(NUM_BOXES - 1):
        x_start = int(BOX_SIZE * (i + 1) + SEP_WIDTH * i)
        x_end = int(x_start + SEP_WIDTH) - 1
        draw.rectangle([x_start, 0, x_end, BOX_SIZE], fill=(128, 128, 128, 255))
    return img, draw


# Below are the functions used for Scatter only


def convert_action_to_img_coordinates(x, y, box):
    """
    Convert (x, y) from the action to the coordinates on the RGB image.

    (cf. `lilgym/envs/README.md` section "Example of a Scatter grid" for example)
    """
    x = x * 20 - box * BOX_SIZE - box * SEP_WIDTH
    y = y * (100 / 5)
    return x, y


def get_item_for_delete_scatter(action, img, img_struct):
    """
    Extract the information of the item to be deleted from the action,
    and convert to the format used to change the image representations.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
    
    Returns:
        img_x (int): x-coordinate (in terms of pixels) on the RGB image
        img_y (int)
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
        box (int): the box the item is in
        box_to_modify (List[Dict]): structured representation of the box
        to modify
    """
    x, y = action.x(), action.y()
    box = get_box(x)
    img_x, img_y = convert_action_to_img_coordinates(x, y, box)

    x_offset = int(BOX_SIZE * box) + SEP_WIDTH * box

    # Find the box
    box_to_modify = img_struct[box]
    return img_x, img_y, x_offset, box, box_to_modify


def can_delete_item_scatter(action, img, img_struct, cell_size=CELL_SIZE):
    """
    Check if the item specified by the action can be deleted (e.g.
    there exist an item on the image that can be removed).

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
        cell_size (int)
    
    Returns:
        (bool)
    """
    x, y, x_offset, box, box_to_modify = get_item_for_delete_scatter(
        action, img, img_struct
    )

    # If there's no element in the box, REMOVE is invalid
    if len(box_to_modify) == 0:
        return False

    # Find the maximum overlapping shape
    item_to_modify_idx = find_largest_item(x, y, cell_size, x_offset, box_to_modify)

    # If no item to be deleted, invalid action
    if item_to_modify_idx == -1:
        return False
    return True


def delete_item_scatter(action, img, img_struct, cell_size=CELL_SIZE):
    """
    (Try to) Delete an item specified by the action, on both
    the structured representation and the PILImage representation of
    the image.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
        cell_size (int)
    
    Returns:
        img_struct (Dict[Dict[Dict]]): modified structured representation
        img (PIL Image)
    """
    x, y, x_offset, box, box_to_modify = get_item_for_delete_scatter(
        action, img, img_struct
    )

    # Check if there's element / get the last element
    if len(box_to_modify) == 0:  # Cannot delete anymore
        return img_struct, draw_on_img(img, img_struct)

    img_struct_for_delete = deepcopy(img_struct)
    curr_color = "Gray"

    # Find the maximum overlapping shape
    item_to_modify_idx = find_largest_item(x, y, cell_size, x_offset, box_to_modify)

    # There is at least one item to be removed
    if item_to_modify_idx != -1:
        item_to_modify = deepcopy(box_to_modify[item_to_modify_idx])

        # Remove the element from the struct, but adding it for deletion
        item_to_modify["color"] = curr_color
        del img_struct[box][item_to_modify_idx]
        img_struct_for_delete[box].append(item_to_modify)
    return img_struct, draw_on_img(img, img_struct_for_delete)


def get_item_for_draw_scatter(action, img, img_struct):
    """
    Extract the information of the item to be added from the action,
    and convert to the format used to change the image representations.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
    
    Returns:
        curr_obj (Dict): structured representation of the item
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
        box (int): the box the item is in
        shapes_in_box (List[shapely.geometry.BaseGeometry]): list of shapes in the current box
    """
    x, y, shape, color, size = (
        action.x(),
        action.y(),
        action.shape(),
        action.color(),
        action.size(),
    )
    box = get_box(x)
    img_x, img_y = convert_action_to_img_coordinates(x, y, box)

    curr_size = Size.int_to_size(size)
    curr_type = Shape.int_to_shape(shape)
    curr_color = Color.int_to_color(color)

    curr_obj = {
        "x_loc": img_x,
        "y_loc": img_y,
        "type": curr_type,
        "color": curr_color,
        "size": curr_size,
    }

    x_offset = int(BOX_SIZE * box) + SEP_WIDTH * box

    # Find the box
    box_to_modify = img_struct[box]

    # shapes_in_box has same order than box_to_modify (items in the box)
    shapes_in_box = get_shapes_in_box(x_offset, box_to_modify)
    return curr_obj, x_offset, box, shapes_in_box


def can_draw_item_scatter(action, img, img_struct, cell_size=CELL_SIZE):
    """
    Check if the item specified by the action can be added (e.g.
    there is no overlapping with an existing item).

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
        cell_size (int)
    
    Returns:
        (bool)
    """
    curr_obj, x_offset, box, shapes_in_box = get_item_for_draw_scatter(
        action, img, img_struct
    )

    valid, conflict_items = check_intersect(shapes_in_box, curr_obj, x_offset)

    if not valid:
        return False

    # Otherwise, try to find a starting point in the cell where the shape can fit
    if conflict_items:
        curr_obj, possible_starting_coordinates = get_drawing_coordinates(
            cell_size, box, shapes_in_box, curr_obj, x_offset
        )
        if not possible_starting_coordinates:
            return False
    return True


def draw_item_scatter(action, img, img_struct, cell_size=CELL_SIZE):
    """
    (Try to) Draw an item specified by the action, on both
    the structured representation and the PILImage representation of
    the image.

    Args:
        action (Type[Action])
        img (PIL Image): image to be modified
        img_struct (Dict[Dict[Dict]]): structured representation of img
        cell_size (int)
    
    Returns:
        img_struct (Dict[Dict[Dict]]): modified structured representation
        img (PIL Image)
    """
    curr_obj, x_offset, box, shapes_in_box = get_item_for_draw_scatter(
        action, img, img_struct
    )

    # If the shape can be drew without affecting other existing shapes, just draw it
    valid, conflict_items = check_intersect(shapes_in_box, curr_obj, x_offset)

    # Ex: drawing a shape over the box
    if not valid:
        # OK, since it means that it's not valid and the curr_obj has not been added
        return img_struct, draw_on_img(img, img_struct)

    # Otherwise, try to find a starting point in the cell where the shape can fit
    if conflict_items:
        curr_obj, possible_starting_coordinates = get_drawing_coordinates(
            cell_size, box, shapes_in_box, curr_obj, x_offset
        )
        if not possible_starting_coordinates:
            return img_struct, draw_on_img(img, img_struct)

    # "Sticky": If the shape is very close to an existing one, stick both
    curr_shape, closest_shape, closest_i, closest_distance = check_closeness(
        box, shapes_in_box, curr_obj, x_offset
    )
    if closest_shape:
        curr_obj = make_sticky(curr_obj, curr_shape, closest_shape, x_offset)

    img_struct[box].append(curr_obj)
    return img_struct, draw_on_img(img, img_struct)


def get_box(x, cell_size=CELL_SIZE):
    """
    Get the box number (0 = left, 1 = middle, 2 = right) according to x.

    Args:
        x: x-coordinate (in terms of pixel) of the item on the RGB image
    
    Returns:
        box (int): the box in which the item is in
    """
    if 0 < x <= BOX_SIZE / cell_size:
        return 0
    elif (
        x > (BOX_SIZE + cell_size * 2) / cell_size
        and x <= (2 * (BOX_SIZE + cell_size)) / cell_size
    ):
        return 1
    elif x > (BOX_SIZE * 2 + cell_size * 4) / cell_size:
        return 2
    return -1


def get_drawing_coordinates(cell_size, box_nb, shapes_in_box, curr_obj, x_offset):
    """
    If there are conflicting items in the cell, try to find a place in the cell 
    where the shape can fit.

    Args:
        cell_size (int)
        box_nb (int)
        shapes_in_box (List[shapely.geometry.BaseGeometry]): list of shapes in the current box
        curr_obj (Dict)
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
    
    Returns:
        curr_obj (Dict): modified curr_obj
        possible_starting_coordinates (bool): whether there exist a place where curr_obj
        can be placed
    """
    possible_starting_coordinates = False

    # 100 is the border of the box
    height = BOX_SIZE
    width = box_nb * BOX_SIZE + (box_nb - 1) * SEP_WIDTH

    # Get curr obj
    x_start = curr_obj["x_loc"]  # no need of the offset since it's in the box already
    y_start = curr_obj["y_loc"]
    # Cell limits
    x_start_limit, x_end_limit = x_start, x_start + cell_size
    y_start_limit, y_end_limit = y_start, y_start + cell_size

    cell_shape = get_shape(
        x_start_limit, y_start_limit, Shape.SQUARE.value, cell_size, x_offset
    )

    x_limit = width - x_offset - curr_obj["size"]
    y_limit = height - curr_obj["size"]

    for x in range(int(x_start_limit), min(int(x_limit), int(x_end_limit))):
        for y in range(int(y_start_limit), min(int(y_limit), int(y_end_limit))):
            curr_shape = get_shape(x, y, curr_obj["type"], curr_obj["size"], x_offset)

            if all(
                not curr_shape.intersection(shape).area > 0.0 for shape in shapes_in_box
            ):
                # Otherwise there can be float rounding error, ex. 0.499999999
                percentage_current_shape = round(
                    curr_shape.intersection(cell_shape).area / curr_shape.area, 2
                )
                percentage_cell = round(
                    curr_shape.intersection(cell_shape).area / cell_shape.area, 2
                )
                if (
                    percentage_current_shape >= OVERLAP_THRES
                    or percentage_cell >= OVERLAP_THRES
                ):
                    # Use the first available starting position
                    curr_obj["x_loc"], curr_obj["y_loc"] = x, y
                    possible_starting_coordinates = True
                    return curr_obj, possible_starting_coordinates

    return curr_obj, possible_starting_coordinates


def check_closeness(
    box_nb,
    shapes_in_box,
    curr_obj,
    x_offset,
    up_thres=ALMOST_TOUCHING_MARGIN,
    low_thres=0,
):
    """
    Check if curr_obj is close enough (given a threshold) to any other objects.

    Args:
        box_nb (int)
        shapes_in_box (List[shapely.geometry.BaseGeometry]): list of shapes in the current box
        curr_obj (Dict)
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
        up_thres (int)
        low_thres (int)
    
    Returns:
        curr_shape (shapely.geometry.BaseGeometry): curr_obj under the shapely representation
        closest_shape (shapely.geometry.BaseGeometry): closest item to curr_obj
        closest_i (int): position of closest_shape in shapes_in_box
        closest_distance (int)
    """
    closest_shape = None
    closest_distance = 200
    if not shapes_in_box:
        return None, closest_shape, -1, closest_distance

    curr_shape = get_shape(
        curr_obj["x_loc"],
        curr_obj["y_loc"],
        curr_obj["type"],
        curr_obj["size"],
        x_offset,
    )

    closest_i = -1
    closest_distance = BOX_SIZE
    for i, shape in enumerate(shapes_in_box):
        _d = int(curr_shape.distance(shape))  # int to avoid diagonal distances
        if low_thres <= _d <= up_thres and _d < closest_distance:
            closest_distance = _d
            closest_shape = shape
            closest_i = i
    return curr_shape, closest_shape, closest_i, closest_distance


def trans_coord(x, closest_nearest_x, curr_nearest_x):
    """
    Translation of the coordinate, to make the corresponding item
    touch the nearest shape.

    Args:
        x (int): x-coordinate (in terms of pixel) on the RGB image
        closest_nearest_x (int)
        curr_nearest_x (int)
    
    Returns:
        (int): the new coordinate for the current item
    """
    if closest_nearest_x == curr_nearest_x:
        return x
    else:
        return x + ((closest_nearest_x) - curr_nearest_x)


def make_sticky(curr_obj, curr_shape, closest_shape, x_offset):
    """
    Heuristics for scatter: ake almost touching objects to touch.

    Args:
        curr_obj (Dict): structured representation of the current item to modify
        curr_shape (shapely.geometry.BaseGeometry): shapely representation of curr_obj
        closest_shape (shapely.geometry.BaseGeometry)
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box

    Returns:
        curr_obj (Dict): modified curr_obj
    """
    # Find where should be touching and where to start drawing the current shape.
    nearest = [o for o in nearest_points(curr_shape, closest_shape)]

    curr_nearest_x, curr_nearest_y = int(nearest[0].x), int(nearest[0].y)
    closest_nearest_x, closest_nearest_y = int(nearest[1].x), int(nearest[1].y)

    curr_ini_x1, curr_ini_y1, curr_ini_x2, curr_ini_y2 = curr_shape.bounds

    curr_new_x1 = trans_coord(curr_ini_x1, closest_nearest_x, curr_nearest_x)
    curr_new_y1 = trans_coord(curr_ini_y1, closest_nearest_y, curr_nearest_y)

    curr_obj["x_loc"] = curr_new_x1 - x_offset  # x coordinate should be within the box
    curr_obj["y_loc"] = curr_new_y1
    return curr_obj


def get_shape(x_loc, y_loc, obj_type, obj_size, x_offset):
    """
    Get the shape under a shapely representation.

    Args:
        x_loc: x_start in the box
        y_loc: y_start in the box
        obj_type: Shape of the item ("circle", "square", "triangle")
        obj_size: Size of the item
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
    
    Returns
        shape (shapely.geometry.BaseGeometry): shapely representation of the object
    """
    x_start = x_offset + x_loc
    y_start = y_loc
    # There's no -1 here, because shapely takes the limits in an inclusive way
    x_end = int(x_start + obj_size / 10 * BOX_SIZE / BOX_OBJECT_RATIO)
    y_end = int(y_start + obj_size / 10 * BOX_SIZE / BOX_OBJECT_RATIO)
    if obj_type == Shape.CIRCLE.value:
        x_center = (x_start + x_end) / 2
        y_center = (y_start + y_end) / 2
        radius = obj_size / 2
        shape = Point(x_center, y_center).buffer(radius)
    elif obj_type == Shape.SQUARE.value:
        shape = Polygon(
            [(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]
        )
    elif obj_type == Shape.TRIANGLE.value:
        x_top = (x_start + x_end) / 2
        shape = Polygon([(x_top, y_start), (x_start, y_end), (x_end, y_end)])
    return shape


def get_shapes_in_box(x_offset, items_in_box):
    """
    Get all the shapes within a box.

    Args:
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
        items_in_box (List[Dict]): list of items in the box

    Returns:
        shapes_in_box (List[shapely.geometry.BaseGeometry]): list of items in the box
        with the shapely representation
    """
    shapes_in_box = []
    for obj_idx, obj in enumerate(items_in_box):
        shapes_in_box.append(
            get_shape(obj["x_loc"], obj["y_loc"], obj["type"], obj["size"], x_offset)
        )
    return shapes_in_box


def check_intersect(shapes_in_box, curr_obj, x_offset):
    """
    Check if there's an intersection between the current item and the items already in the box.
    
    Args:
        shapes_in_box (List[shapely.geometry.BaseGeometry]): list of items in the box
        with the shapely representation
        curr_obj (Dict): item to be added
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
    
    Returns:
        valid (bool): whether it's possible to place the current item
        conflict_items (List[int]): list of items that would overlap with the current item
    """
    conflict_items = []
    valid = True

    x_start = x_offset + curr_obj["x_loc"]
    y_start = curr_obj["y_loc"]
    x_end = int(x_start + curr_obj["size"] / 10 * BOX_SIZE / BOX_OBJECT_RATIO) - 1
    y_end = int(y_start + curr_obj["size"] / 10 * BOX_SIZE / BOX_OBJECT_RATIO) - 1

    # If the shape goes outside of the box
    if x_end >= x_offset + BOX_SIZE or y_end >= BOX_SIZE:
        valid = False
        return valid, conflict_items

    curr_shape = get_shape(
        curr_obj["x_loc"],
        curr_obj["y_loc"],
        curr_obj["type"],
        curr_obj["size"],
        x_offset,
    )

    # Check intersections
    for obj_idx, shape in enumerate(shapes_in_box):
        inter_area = shape.intersection(curr_shape).area
        if inter_area > 0:
            conflict_items.append(obj_idx)
    return valid, conflict_items


def find_largest_item(x, y, cell_size, x_offset, items_in_box):
    """
    Given the cell (x, y), find the shape with the largest area
    intersecting with the cell.
    
    Args:
        x (int): x-coordinate (in terms of pixel) of the current cell on the RGB image
        y (int): y-coordinate (in terms of pixel) of the current cell on the RGB image
        cell_size (int)
        x_offset (int): offset from the leftmost pixel of the image to 
        the current box
        items_in_box (List[Dict]): list of items in the box
    
    Returns:
        max_shape_idx (int): index of the largest item overlapping with the cell
    """

    # Create the cell object
    b_coord = {
        "x1": x,
        "y1": y,
        "x2": min(x + cell_size, BOX_SIZE),
        "y2": min(y + cell_size, BOX_SIZE),
    }
    box = Polygon(
        [
            (b_coord["x1"], b_coord["y1"]),
            (b_coord["x2"], b_coord["y1"]),
            (b_coord["x2"], b_coord["y2"]),
            (b_coord["x1"], b_coord["y2"]),
        ]
    )

    # Variables keeping track of the maximum overlapping
    max_intersection = 0.0
    max_shape_idx = -1

    for obj_idx, obj in enumerate(items_in_box):
        shape = get_shape(
            obj["x_loc"], obj["y_loc"], obj["type"], obj["size"], x_offset
        )
        inter_area = shape.intersection(box).area
        if inter_area > 0.0 and inter_area > max_intersection:
            max_intersection = inter_area
            max_shape_idx = obj_idx
    return max_shape_idx
