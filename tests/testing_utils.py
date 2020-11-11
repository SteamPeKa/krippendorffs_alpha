# coding=utf-8
# Creation date: 28 окт. 2020
# Creation time: 19:16
# Creator: SteamPeKa

from typing import Collection, Hashable, Union

import numpy
import pytest


def assert_equal_tensors(expected: numpy.ndarray, actual: numpy.ndarray, additional_string: Union[None, str] = None):
    assert expected.shape == actual.shape, f"Tensors have different shapes: {expected.shape} and {actual.shape}"
    abs_difference = numpy.max(numpy.abs(expected - actual))
    fail_message = (
        f"Tensors are not equal with maximum abs difference of {abs_difference}.\n"
        f"Expected:\n"
        f"{expected}\n"
        f"Actual:\n"
        f"{actual}\n"
        f"Difference:\n"
        f"{expected - actual}"
    )
    if additional_string is not None:
        fail_message += "\n####\n" + additional_string + "\n####"
    assert abs_difference == pytest.approx(0), fail_message


def assert_collections_equal_as_sets(expected: Collection[Hashable], actual: Collection[Hashable]):
    expected_set = set(expected)
    actual_set = set(actual)
    sym_diff = expected_set.symmetric_difference(actual_set)
    assert len(sym_diff) == 0, (f"Collections are not equal.\n"
                                f"Expected collection-set: {expected_set} (of size {len(expected_set)})\n"
                                f"Actual collection-set: {actual_set} (of size {len(actual_set)})\n"
                                f"Symmetric difference: {sym_diff} (of length {len(sym_diff)} != 0)")
