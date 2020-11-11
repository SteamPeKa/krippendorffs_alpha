# coding=utf-8
# Creation date: 27 окт. 2020
# Creation time: 18:52
# Creator: SteamPeKa

import csv
import os

import numpy
import pytest

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


class TestMakeCoincidencesMatrixFromDataMatrix(object):
    def test_data_from_example_no_omit(self):
        expected_value_by_unit_matrix = numpy.array([
            [3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2, 0],
            [0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0, 0],
            [0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        ])

        actual_value_by_unit_matrix = krippendorffs_alpha._calculation._make_value_by_unit_matrix_from_data_matrix(
            DATA_MATRIX, omit_unpairable=False
        )
        testing_utils.assert_equal_tensors(expected_value_by_unit_matrix, actual_value_by_unit_matrix)

    def test_data_from_example_omit_unpairable(self):
        expected_value_by_unit_matrix = numpy.array([
            [3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2],
            [0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0],
            [0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        ])

        actual_value_by_unit_matrix = krippendorffs_alpha._calculation._make_value_by_unit_matrix_from_data_matrix(
            DATA_MATRIX, omit_unpairable=True
        )
        testing_utils.assert_equal_tensors(expected_value_by_unit_matrix, actual_value_by_unit_matrix)


# noinspection PyPep8Naming
class Test_CalcAlpha(object):

    def test_e_nominal(self):
        metric_tensor = krippendorffs_alpha.metrics.NominalMetric().get_metric_tensor(list(range(1, 6)),
                                                                                      symmetric=False)
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha(DATA_MATRIX, metric_tensor)
        assert actual_alpha == pytest.approx(0.743, abs=0.001)

    def test_e_interval(self):
        metric_tensor = krippendorffs_alpha.metrics.IntervalMetric().get_metric_tensor(list(range(1, 6)),
                                                                                       symmetric=False)
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha(DATA_MATRIX, metric_tensor)
        assert actual_alpha == pytest.approx(0.849, abs=0.001)


class TestCalcAlpha(object):
    def test_e_nominal(self):
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = csv.reader(f, delimiter="\t")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "NULL" else None
            )

        testing_utils.assert_equal_tensors(DATA_MATRIX, prepared_data.answers_tensor)
        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_data, "nominal")
        assert actual_alpha == pytest.approx(0.743, abs=0.001)

    def test_e_interval(self):
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = csv.reader(f, delimiter="\t")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "NULL" else None
            )

        testing_utils.assert_equal_tensors(DATA_MATRIX, prepared_data.answers_tensor)
        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_data, "interval")
        assert actual_alpha == pytest.approx(0.849, abs=0.001)

    def test_wikipedia_nominal(self):
        with open(os.path.join("tests", "example_wikipedia.csv"), "r") as f:
            input_table = csv.reader(f, delimiter=",")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "*" else None
            )

        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_data, "nominal")
        assert actual_alpha == pytest.approx(0.691, abs=0.001)

    def test_wikipedia_interval(self):
        with open(os.path.join("tests", "example_wikipedia.csv"), "r") as f:
            input_table = csv.reader(f, delimiter=",")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "*" else None
            )

        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_data, "interval")
        assert actual_alpha == pytest.approx(0.811, abs=0.001)


class Test_CalcAlphaByPrecomputes(object):
    def test_e_nominal(self):
        metric_name = "nominal"
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = csv.reader(f, delimiter="\t")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "NULL" else None
            )

        (assignment_matrix,
         full_cross_disagreement_tensor) = krippendorffs_alpha._calculation._prepare_bootstrap_precomputes(
            prepared_data=prepared_data,
            metric=metric_name
        )
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha_by_precomputes(
            assignment_matrix=assignment_matrix,
            full_cross_disagreement_tensor=full_cross_disagreement_tensor
        )
        assert actual_alpha == pytest.approx(0.743, abs=0.001)

    def test_e_interval(self):
        metric_name = "interval"
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = csv.reader(f, delimiter="\t")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "NULL" else None
            )

        (assignment_matrix,
         full_cross_disagreement_tensor) = krippendorffs_alpha._calculation._prepare_bootstrap_precomputes(
            prepared_data=prepared_data,
            metric=metric_name
        )
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha_by_precomputes(
            assignment_matrix=assignment_matrix,
            full_cross_disagreement_tensor=full_cross_disagreement_tensor
        )
        assert actual_alpha == pytest.approx(0.849, abs=0.001)

    def test_wikipedia_nominal(self):
        metric_name = "nominal"
        with open(os.path.join("tests", "example_wikipedia.csv"), "r") as f:
            input_table = csv.reader(f, delimiter=",")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "*" else None
            )

        (assignment_matrix,
         full_cross_disagreement_tensor) = krippendorffs_alpha._calculation._prepare_bootstrap_precomputes(
            prepared_data=prepared_data,
            metric=metric_name
        )
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha_by_precomputes(
            assignment_matrix=assignment_matrix,
            full_cross_disagreement_tensor=full_cross_disagreement_tensor
        )
        assert actual_alpha == pytest.approx(0.691, abs=0.001)

    def test_wikipedia_interval(self):
        metric_name = "interval"
        with open(os.path.join("tests", "example_wikipedia.csv"), "r") as f:
            input_table = csv.reader(f, delimiter=",")
            prepared_data = krippendorffs_alpha.data_converters.from_list_of_lists(
                input_table=input_table,
                header=True,
                row_legend=True,
                upper_level="observer",
                value_constructor=lambda s: int(s.strip()) if s.strip() != "*" else None
            )

        (assignment_matrix,
         full_cross_disagreement_tensor) = krippendorffs_alpha._calculation._prepare_bootstrap_precomputes(
            prepared_data=prepared_data,
            metric=metric_name
        )
        actual_alpha = krippendorffs_alpha._calculation._calc_alpha_by_precomputes(
            assignment_matrix=assignment_matrix,
            full_cross_disagreement_tensor=full_cross_disagreement_tensor
        )
        assert actual_alpha == pytest.approx(0.811, abs=0.001)
