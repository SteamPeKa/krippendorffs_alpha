# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 10:48
# Creator: SteamPeKa

from ._abstract_example import AbstractExample
from .examples_from_article import example_A, example_B, example_C
from .multiple_assignments_examples import multiple_assignments_example_A
from .wikipedia_example import wikipedia_example_A

all_examples = {
    "A": {
        "example": example_A,
        "metrics": ["nominal"]
    },
    "B": {
        "example": example_B,
        "metrics": ["nominal", "interval", "ratio"]
    },
    "C": {
        "example": example_C,
        "metrics": ["nominal", "interval", "ratio"],
    },
    "Multiple assignments": {
        "example": multiple_assignments_example_A,
        "metrics": ["nominal", "interval", "ratio"]
    },
    "From Wikipedia ": {
        "example": wikipedia_example_A,
        "metrics": ["nominal", "interval", "ratio"]
    }
}
