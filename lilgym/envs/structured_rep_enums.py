#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Classes specifying the properties of the items on the image
# Based on the code of Weakly Supervised Semantic Parsing with Abstract Examples, Goldman et al., 2019.


from enum import Enum


class Size(Enum):
    SMALL = 10
    MEDIUM = 20
    BIG = 30

    def as_int(self):
        if self == Size.SMALL:
            return 0
        elif self == Size.MEDIUM:
            return 1
        elif self == Size.LARGE:
            return 2

    def int_to_size(number):
        if number == 0:
            return 10
        elif number == 1:
            return 20
        elif number == 2:
            return 30

    @classmethod
    def int_to_str(cls, number):
        if number == 0:
            return "SMALL"
        elif number == 1:
            return "MEDIUM"
        elif number == 2:
            return "LARGE"

    @classmethod
    def str_to_int(cls, size: str):
        if size == "SMALL":
            return 0
        elif size == "MEDIUM":
            return 1
        elif size == "LARGE":
            return 2


class Color(Enum):
    YELLOW = "Yellow"
    BLACK = "Black"
    BLUE = "#0099ff"
    GRAY = "Gray"

    def as_rgb(self):
        if self == Color.YELLOW:
            return 250, 255, 0, 255
        elif self == Color.BLACK:
            return 0, 0, 0, 255
        elif self == Color.BLUE:
            return 15, 131, 255, 255
        elif self == Color.GRAY:
            return 211, 211, 211, 255

    def as_hex(self):
        if self == Color.YELLOW:
            return "#FFFF0C"
        elif self == Color.BLACK:
            return "#000000"
        elif self == Color.BLUE:
            return "#0099ff"
        elif self == Color.GRAY:
            return "#d3d3d3"

    def as_int(self):
        if self == Color.YELLOW:
            return 0
        elif self == Color.BLACK:
            return 1
        elif self == Color.BLUE:
            return 2

    @classmethod
    def int_to_color(cls, number):
        if number == 0:
            return cls.YELLOW.value
        elif number == 1:
            return cls.BLACK.value
        elif number == 2:
            return cls.BLUE.value

    @classmethod
    def int_to_str(cls, number):
        if number == 0:
            return "YELLOW"
        elif number == 1:
            return "BLACK"
        elif number == 2:
            return "BLUE"

    @classmethod
    def str_to_int(cls, color: str):
        if color == "YELLOW":
            return 0
        elif color == "BLACK":
            return 1
        elif color == "BLUE":
            return 2


class Shape(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"

    def as_int(self):
        if self == Shape.CIRCLE:
            return 0
        elif self == Shape.SQUARE:
            return 1
        elif self == Shape.TRIANGLE:
            return 2

    @classmethod
    def int_to_shape(cls, number):
        if number == 0:
            return cls.CIRCLE.value
        elif number == 1:
            return cls.SQUARE.value
        elif number == 2:
            return cls.TRIANGLE.value

    @classmethod
    def int_to_str(cls, number):
        if number == 0:
            return "CIRCLE"
        elif number == 1:
            return "SQUARE"
        elif number == 2:
            return "TRIANGLE"

    @classmethod
    def str_to_int(cls, shape: str):
        if shape == "CIRCLE":
            return 0
        elif shape == "SQUARE":
            return 1
        elif shape == "TRIANGLE":
            return 2


class TowerBox(Enum):
    LEFT = "LEFT"
    MIDDLE = "MIDDLE"
    RIGHT = "RIGHT"

    @classmethod
    def int_to_str(cls, number):
        if number == 0:
            return cls.LEFT.value
        elif number == 1:
            return cls.MIDDLE.value
        elif number == 2:
            return cls.RIGHT.value

    @classmethod
    def str_to_int(cls, box: str):
        if box == TowerBox.LEFT.value:
            return 0
        elif box == TowerBox.MIDDLE.value:
            return 1
        elif box == TowerBox.RIGHT.value:
            return 2


class Location(Enum):
    TOP = "top"
    SECOND = "second"
    BOTTOM = "bottom"


class Relation(Enum):
    ABOVE = "above"
    BELOW = "below"
    TOUCH = "touch"
    CLOSELY_TOUCH = "closely touch"


class Side(Enum):
    RIGHT = ("right",)
    LEFT = ("left",)
    TOP = ("top",)
    BOTTOM = "bottom"
    ANY = "any"
