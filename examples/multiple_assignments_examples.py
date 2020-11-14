# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 11:26
# Creator: SteamPeKa
from . import AbstractExample

_DATA = {
    "OBSERVER A": {
        "U2": "Z",
        "U3": "V",
        "U5": "V",
        "U8": "X",
        "U9": "Y",
        "U10": "X",
        "V1": [
            "Y",
            "Z",
            "Z"
        ],
        "V2": [
            "W",
            "W",
            "Y"
        ]
    },
    "OBSERVER B": {
        "U1": "Y",
        "U5": "V",
        "U6": "Y",
        "U7": "Z",
        "U9": "Y",
        "U10": "X",
        "V1": [
            "W",
            "Z",
            "Y"
        ],
        "V2": [
            "W",
            "W",
            "X"
        ]
    },
    "OBSERVER C": {
        "U1": "X",
        "U2": "Z",
        "U4": "Y",
        "U6": "Z",
        "U8": "X",
        "U10": "W",
        "V1": [
            "Z",
            "Y",
            "Y"
        ],
        "V2": [
            "W",
            "W",
            "V"
        ]
    },
    "OBSERVER D": {
        "U1": "X",
        "U3": "V",
        "U4": "X",
        "U5": "W",
        "U7": "Y",
        "U8": "Y",
        "V1": [
            "X",
            "Y",
            "Y"
        ],
        "V2": [
            "X",
            "W",
            "W"
        ]
    },
    "OBSERVER E": {
        "U2": "X",
        "U3": "Z",
        "U4": "V",
        "U6": "V",
        "U7": "Y",
        "U9": "V",
        "V1": [
            "X",
            "V",
            "Y"
        ],
        "V2": [
            "Y",
            "Z",
            "V"
        ]
    }
}

multiple_assignments_example_A = AbstractExample(_DATA,
                                                 ["OBSERVER A", "OBSERVER B", "OBSERVER C", "OBSERVER D", "OBSERVER E"],
                                                 ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10",
                                                  "V1", "V2"],
                                                 ["V", "W", "X", "Y", "Z"])
