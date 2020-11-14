# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 10:49
# Creator: SteamPeKa
from . import AbstractExample

"""
Examples from
Krippendorff K. Computing Krippendorff's alpha-reliability. – 2011.
(https://repository.upenn.edu/cgi/viewcontent.cgi?article=1043&context=asc_papers)
"""

"""
Example from section A
"""
example_A = AbstractExample({
    "Meg": {
        "1": "0", "2": "1", "3": "0", "4": "0", "5": "0", "6": "0", "7": "0", "8": "0", "9": "1", "10": "0",
    },
    "Owen": {
        "1": "1", "2": "1", "3": "1", "4": "0", "5": "0", "6": "1", "7": "0", "8": "0", "9": "0", "10": "0",
    }
},
    ["Meg", "Owen"],
    [str(v) for v in range(1, 11)],
    ["0", "1"])

example_B = AbstractExample({
    "Ben": {"1": "a", "2": "a", "3": "b", "4": "b", "5": "d", "6": "c",
            "7": "c", "8": "c", "9": "e", "10": "d", "11": "d", "12": "a"},
    "Gerry": {"1": "b", "2": "a", "3": "b", "4": "b", "5": "b", "6": "c",
              "7": "c", "8": "c", "9": "e", "10": "d", "11": "d", "12": "d"}},
    ["Ben", "Gerry"],
    [str(v) for v in range(1, 13)],
    ["a", "b", "c", "d", "e"]
)

__EXAMPLE_C_DATA = {
    "OBSERVER A": {
        "UNIT 1": "1",
        "UNIT 2": "2",
        "UNIT 3": "3",
        "UNIT 4": "3",
        "UNIT 5": "2",
        "UNIT 6": "1",
        "UNIT 7": "4",
        "UNIT 8": "1",
        "UNIT 9": "2"
    },
    "OBSERVER B": {
        "UNIT 1": "1",
        "UNIT 10": "5",
        "UNIT 12": "3",
        "UNIT 2": "2",
        "UNIT 3": "3",
        "UNIT 4": "3",
        "UNIT 5": "2",
        "UNIT 6": "2",
        "UNIT 7": "4",
        "UNIT 8": "1",
        "UNIT 9": "2"
    },
    "OBSERVER C": {
        "UNIT 10": "5",
        "UNIT 11": "1",
        "UNIT 2": "3",
        "UNIT 3": "3",
        "UNIT 4": "3",
        "UNIT 5": "2",
        "UNIT 6": "3",
        "UNIT 7": "4",
        "UNIT 8": "2",
        "UNIT 9": "2"
    },
    "OBSERVER D": {
        "UNIT 1": "1",
        "UNIT 10": "5",
        "UNIT 11": "1",
        "UNIT 2": "2",
        "UNIT 3": "3",
        "UNIT 4": "3",
        "UNIT 5": "2",
        "UNIT 6": "4",
        "UNIT 7": "4",
        "UNIT 8": "1",
        "UNIT 9": "2"
    }
}
example_C = AbstractExample(__EXAMPLE_C_DATA,
                            ["OBSERVER A", "OBSERVER B", "OBSERVER C", "OBSERVER D"],
                            ["UNIT {:d}".format(v) for v in range(1, 13)],
                            [str(v) for v in range(1, 6)])
