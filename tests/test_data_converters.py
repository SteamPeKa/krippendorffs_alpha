# coding=utf-8
# Creation date: 28 окт. 2020
# Creation time: 19:33
# Creator: SteamPeKa

import json
import os

import numpy

import krippendorffs_alpha
import testing_utils

# Example E data matrix
OBSERVER_A_DATA = numpy.array([
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]).T
OBSERVER_B_DATA = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
]).T
OBSERVER_C_DATA = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
]).T
OBSERVER_D_DATA = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
]).T

DATA_MATRIX = numpy.zeros((4, 12, 5))
DATA_MATRIX[0, :, :] = OBSERVER_A_DATA
DATA_MATRIX[1, :, :] = OBSERVER_B_DATA
DATA_MATRIX[2, :, :] = OBSERVER_C_DATA
DATA_MATRIX[3, :, :] = OBSERVER_D_DATA

DATA_MATRIX.flags.writeable = False
OBSERVER_A_DATA.flags.writeable = False
OBSERVER_B_DATA.flags.writeable = False
OBSERVER_C_DATA.flags.writeable = False
OBSERVER_D_DATA.flags.writeable = False

POSSIBLE_VALUES = ["1", "2", "3", "4", "5"]

PERMUTATIONS_LIMIT = 10

CHOSEN_PERMUTATIONS = {(
    tuple(range(DATA_MATRIX.shape[0])),
    tuple(range(DATA_MATRIX.shape[1]))
)}
while len(CHOSEN_PERMUTATIONS) < PERMUTATIONS_LIMIT:
    CHOSEN_PERMUTATIONS.add((
        tuple(numpy.random.permutation(tuple(range(DATA_MATRIX.shape[0])))),
        tuple(numpy.random.permutation(tuple(range(DATA_MATRIX.shape[1]))))
    ))


def decode_answer_array(answer_array: numpy.ndarray, constructor=None):
    assert len(answer_array.shape) == 1
    result = None
    for index, value in enumerate(answer_array):
        if value == 0.0:
            assert result is None or result != index + 1
        elif value == 1.0:
            assert result is None
            result = index + 1
    if result is None:
        return result
    elif constructor is None:
        return result
    else:
        return constructor(result)


def generate_permutations(upper_level="observer", constructor=False):
    assert upper_level in {"observer", "unit"}
    for observers_permutation, units_permutation in CHOSEN_PERMUTATIONS:
        expected_data_matrix = DATA_MATRIX[observers_permutation, :, :]
        expected_data_matrix = expected_data_matrix[:, units_permutation, :]
        if upper_level == "observer":
            input_table = expected_data_matrix.copy()
        elif upper_level == "unit":
            input_table = expected_data_matrix.transpose([1, 0, 2])
        else:
            raise NotImplementedError()
        input_table = [[decode_answer_array(encoded_value, constructor=int if constructor else str)
                        for encoded_value in row]
                       for row in input_table]
        yield input_table, expected_data_matrix


def generate_data(upper_level="observer", constructor: bool = False,
                  header: bool = False, row_legend: bool = False):
    assert upper_level in {"observer", "unit"}
    for input_data, expected_result in generate_permutations(upper_level=upper_level, constructor=constructor):
        if row_legend:
            if upper_level == "observer":
                input_data = [["OBSERVER_{}".format(i)] + row for i, row in enumerate(input_data)]
            elif upper_level == "unit":
                input_data = [["UNIT_{}".format(i)] + row for i, row in enumerate(input_data)]

        if header:
            first_row = None
            if upper_level == "observer":
                first_row = ["UNIT_{}".format(i) for i in range(len(input_data[0]))]
            elif upper_level == "unit":
                first_row = ["OBSERVER_{}".format(i) for i in range(len(input_data[0]))]
            assert first_row is not None
            if row_legend:
                # noinspection PyTypeChecker
                first_row = [None] + first_row[:-1]
            input_data = [first_row] + input_data
        yield input_data, expected_result


class TestFromListOfLists(object):

    @classmethod
    def routine(cls, upper_level, constructor, header, row_legend, possible_values):
        if header == "make":
            if upper_level == "observer":
                header = tuple("GH_UNIT_{}".format(i) for i in range(DATA_MATRIX.shape[1]))
            elif upper_level == "unit":
                header = tuple("GH_OBSERVER_{}".format(i) for i in range(DATA_MATRIX.shape[0]))
            else:
                raise NotImplementedError()
        if row_legend == "make":
            if upper_level == "observer":
                row_legend = tuple("GH_OBSERVER_{}".format(i) for i in range(DATA_MATRIX.shape[0]))
            elif upper_level == "unit":
                row_legend = tuple("GH_UNIT_{}".format(i) for i in range(DATA_MATRIX.shape[1]))
            else:
                raise NotImplementedError()
        permutations_generator = generate_data(upper_level=upper_level,
                                               constructor=constructor is not None,
                                               header=isinstance(header, bool) and header is True,
                                               row_legend=isinstance(row_legend, bool) and row_legend is True)
        expected_values = POSSIBLE_VALUES
        if constructor is not None:
            # noinspection PyCallingNonCallable
            expected_values = [constructor(v) for v in expected_values]
        for input_table, expected_result in permutations_generator:
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(input_table=input_table,
                                                                                   header=header,
                                                                                   row_legend=row_legend,
                                                                                   upper_level=upper_level,
                                                                                   value_constructor=constructor,
                                                                                   possible_values=possible_values, )
            testing_utils.assert_collections_equal_as_sets(expected_values, prepared_data.possible_values)
            if possible_values is not None:
                assert all(a == b
                           for a, b in
                           zip(possible_values, prepared_data.possible_values))  # checking the initial order
            values_permutation = tuple(expected_values.index(new_index) for new_index in prepared_data.possible_values)
            expected_result = expected_result[:, :, values_permutation]

            testing_utils.assert_equal_tensors(expected_result, prepared_data.answers_tensor)

            if isinstance(header, bool):
                if header:
                    if isinstance(row_legend, bool) and row_legend:
                        right_header = tuple(input_table[0][1:])
                    else:
                        right_header = tuple(input_table[0])
                else:
                    if isinstance(row_legend, bool) and row_legend:
                        right_header = tuple(range(len(input_table[0][1:])))
                    else:
                        right_header = tuple(range(len(input_table[0])))
            else:
                # noinspection PyTypeChecker
                right_header = tuple(header)

            if isinstance(row_legend, bool):
                if row_legend:
                    if isinstance(header, bool) and header:
                        right_row_legend = tuple(row[0] for row in input_table[1:])
                    else:
                        right_row_legend = tuple(row[0] for row in input_table)
                else:
                    if isinstance(header, bool) and header:
                        right_row_legend = tuple(range(len(input_table) - 1))
                    else:
                        right_row_legend = tuple(range(len(input_table)))
            else:
                # noinspection PyTypeChecker
                right_row_legend = tuple(row_legend)

            if upper_level == "observer":
                assert right_row_legend == prepared_data.observers_names
                assert right_header == prepared_data.units_names
            elif upper_level == "unit":
                assert right_row_legend == prepared_data.units_names
                assert right_header == prepared_data.observers_names

    def test_observer_upper_level_str_value_no_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_str_value_no_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_no_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_no_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_str_value_no_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_no_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_no_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_no_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_first_row_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_str_value_first_row_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_first_row_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_first_row_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_str_value_first_row_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_first_row_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_first_row_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_first_row_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_generate_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_str_value_generate_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_generate_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_generate_header_no_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=None)

    def test_observer_upper_level_str_value_generate_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_generate_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_generate_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_generate_header_no_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_no_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_str_value_no_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_no_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_no_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_str_value_no_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_no_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_no_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_no_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_first_row_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_str_value_first_row_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_first_row_header_first_col_row_legend_no_possible_values_specified(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_first_row_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_str_value_first_row_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_first_row_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_first_row_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_first_row_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_generate_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_str_value_generate_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_int_constructor_generate_header_first_col_row_legend_no_possible_values_specified(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=None)

    def test_unit_upper_level_int_constructor_generate_header_first_col_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=None)

    def test_observer_upper_level_str_value_generate_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_generate_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_generate_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_generate_header_first_col_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_no_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_str_value_no_header_no_row_legend_generate_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_int_constructor_no_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_int_constructor_no_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_str_value_no_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_no_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_no_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_no_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_first_row_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_str_value_first_row_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_int_constructor_first_row_header_generate_row_legend_no_possible_values_specified(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_int_constructor_first_row_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_str_value_first_row_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_first_row_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_first_row_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_first_row_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_generate_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_str_value_generate_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_int_constructor_generate_header_generate_row_legend_no_possible_values_specified(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=None)

    def test_unit_upper_level_int_constructor_generate_header_generate_row_legend_no_possible_values_specified(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=None)

    def test_observer_upper_level_str_value_generate_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_unit_upper_level_str_value_generate_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=["1", "2", "3", "4", "5"])

    def test_observer_upper_level_int_constructor_generate_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_unit_upper_level_int_constructor_generate_header_generate_row_legend_with_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=[1, 2, 3, 4, 5])

    def test_observer_upper_level_str_value_no_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_no_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_no_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_no_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_first_row_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_first_row_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_first_row_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_first_row_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_generate_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_generate_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=False,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_generate_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_generate_header_no_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=False,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_no_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_no_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_no_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_no_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_first_row_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_first_row_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_first_row_header_first_col_row_legend_with_possible_values_order_2(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_first_row_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_generate_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_generate_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend=True,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_generate_header_first_col_row_legend_with_possible_values_order_2(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_generate_header_first_col_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend=True,
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_no_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_no_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=False,
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_no_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_no_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=False,
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_first_row_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_first_row_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header=True,
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_first_row_header_generate_row_legend_with_possible_values_order_2(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_first_row_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header=True,
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])

    def test_observer_upper_level_str_value_generate_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_unit_upper_level_str_value_generate_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     header="make",
                     row_legend="make",
                     possible_values=["5", "4", "3", "2", "1"])

    def test_observer_upper_level_int_constructor_generate_header_generate_row_legend_with_possible_values_order_2(
            self):
        self.routine(upper_level="observer",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])

    def test_unit_upper_level_int_constructor_generate_header_generate_row_legend_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     header="make",
                     row_legend="make",
                     possible_values=[5, 4, 3, 2, 1])


class TestFromDictOfDicts(object):
    with open(os.path.join("tests", "example_E_data.json"), "r") as f:
        __input_data = json.load(f)

    @classmethod
    def routine(cls, upper_level, constructor, possible_values):
        if possible_values is not None:
            if constructor is not None:
                possible_values = [constructor(v) for v in possible_values]
        expected_values = set()
        expected_units = set()
        expected_observers = set()

        input_table = {}
        if upper_level == "observer":
            for observer_id, observers_answers in cls.__input_data.items():
                expected_observers.add(observer_id)
                for unit_id, answer in observers_answers.items():
                    expected_units.add(unit_id)
                    assert input_table.setdefault(observer_id, {}).setdefault(unit_id, answer) == answer
                    expected_values.add(answer)
        elif upper_level == "unit":
            for observer_id, observers_answers in cls.__input_data.items():
                expected_observers.add(observer_id)
                for unit_id, answer in observers_answers.items():
                    expected_units.add(unit_id)
                    assert input_table.setdefault(unit_id, {}).setdefault(observer_id, answer) == answer
                    expected_values.add(answer)
        else:
            raise NotImplementedError(f"Unknown upper_level: {upper_level}")

        testing_utils.assert_collections_equal_as_sets(expected_values, {"1", "2", "3", "4", "5"})
        expected_values = tuple(POSSIBLE_VALUES)
        if constructor is not None:
            expected_values = tuple(constructor(v) for v in expected_values)

        testing_utils.assert_collections_equal_as_sets(expected_units,
                                                       {"UNIT 1", "UNIT 2", "UNIT 3", "UNIT 4", "UNIT 5", "UNIT 6",
                                                        "UNIT 7", "UNIT 8", "UNIT 9", "UNIT 10", "UNIT 11", "UNIT 12"})
        expected_units = ("UNIT 1", "UNIT 2", "UNIT 3", "UNIT 4", "UNIT 5", "UNIT 6",
                          "UNIT 7", "UNIT 8", "UNIT 9", "UNIT 10", "UNIT 11", "UNIT 12")

        testing_utils.assert_collections_equal_as_sets(expected_observers, ("OBSERVER A", "OBSERVER B",
                                                                            "OBSERVER C", "OBSERVER D"))
        expected_observers = ("OBSERVER A", "OBSERVER B", "OBSERVER C", "OBSERVER D")

        if constructor is not None:
            # noinspection PyCallingNonCallable
            expected_values = [constructor(v) for v in expected_values]

        prepared_data = krippendorffs_alpha.data_converters.from_dict_of_dicts(input_table=input_table,
                                                                               upper_level=upper_level,
                                                                               value_constructor=constructor,
                                                                               possible_values=possible_values)
        expected_result = DATA_MATRIX.copy()

        testing_utils.assert_collections_equal_as_sets(expected_observers, prepared_data.observers_names)
        observers_permutation = tuple(expected_observers.index(new_name)
                                      for new_name in prepared_data.observers_names)
        expected_result = expected_result[observers_permutation, :, :]

        testing_utils.assert_collections_equal_as_sets(expected_units, prepared_data.units_names)
        units_permutation = tuple(expected_units.index(new_index) for new_index in prepared_data.units_names)
        expected_result = expected_result[:, units_permutation, :]

        testing_utils.assert_collections_equal_as_sets(expected_values, prepared_data.possible_values)
        if possible_values is not None:
            assert all(a == b
                       for a, b in zip(possible_values, prepared_data.possible_values))  # checking the initial order
        values_permutation = tuple(expected_values.index(new_index) for new_index in prepared_data.possible_values)
        expected_result = expected_result[:, :, values_permutation]

        testing_utils.assert_equal_tensors(expected_result, prepared_data.answers_tensor,
                                           additional_string=f"{expected_observers}\n"
                                                             f"{prepared_data.observers_names}\n"
                                                             f"{observers_permutation}\n\n"
                                                             f"{expected_units}\n"
                                                             f"{prepared_data.units_names}\n"
                                                             f"{units_permutation}\n\n"
                                                             f"{expected_values}\n"
                                                             f"{prepared_data.possible_values}\n"
                                                             f"{possible_values}\n"
                                                             f"{values_permutation}")

    def test_upper_observer_no_constructor_no_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     possible_values=None)

    def test_upper_unit_no_constructor_no_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     possible_values=None)

    def test_upper_observer_int_constructor_no_possible_values(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     possible_values=None)

    def test_upper_unit_int_constructor_no_possible_values(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     possible_values=None)

    def test_upper_observer_no_constructor_with_possible_values_order_1(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_upper_observer_no_constructor_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=None,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_upper_unit_no_constructor_with_possible_values_order_1(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_upper_unit_no_constructor_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=None,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_upper_observer_int_constructor_with_possible_values_order_1(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_upper_observer_int_constructor_with_possible_values_order_2(self):
        self.routine(upper_level="observer",
                     constructor=int,
                     possible_values=["5", "4", "3", "2", "1"])

    def test_upper_unit_int_constructor_with_possible_values_order_1(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     possible_values=["1", "2", "3", "4", "5"])

    def test_upper_unit_int_constructor_with_possible_values_order_2(self):
        self.routine(upper_level="unit",
                     constructor=int,
                     possible_values=["5", "4", "3", "2", "1"])
