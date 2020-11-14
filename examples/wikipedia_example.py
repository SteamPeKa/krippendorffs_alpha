# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 21:10
# Creator: SteamPeKa


from ._abstract_example import AbstractExample

wikipedia_example_A = AbstractExample(
    {
        "Coder A": {"6": "3", "7": "4", "8": "1", "9": "2", "10": "1", "11": "1", "12": "3", "13": "3", "15": "3"},
        "Coder B": {"1": "1", "3": "2", "4": "1", "5": "3", "6": "3", "7": "4", "8": "3"},
        "Coder C": {"3": "2", "4": "1", "5": "3", "6": "4", "7": "4", "9": "2", "10": "1", "11": "1", "12": "3",
                    "13": "3", "15": "4"},
    },
    ["Coder A", "Coder B", "Coder C"],
    [str(v) for v in range(1, 16) if v not in {2, 14}],
    [str(v) for v in range(1, 5)]
)
