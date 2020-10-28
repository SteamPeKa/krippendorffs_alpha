# coding=utf-8
# Creation date: 27 окт. 2020
# Creation time: 18:52
# Creator: SteamPeKa

import numpy
import krippendorffs_alpha
import pytest

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


class TestGetSummingBounds(object):
    def test_immutable(self):
        bounds = krippendorffs_alpha._calculation.get_summing_bounds(3)
        with pytest.raises(ValueError) as exception_info:
            bounds[0, 0] = 10
        assert exception_info.type is ValueError

    def test_squared(self):
        for size in range(1, 11):
            bounds = krippendorffs_alpha._calculation.get_summing_bounds(size)
            assert len(bounds.shape) == 2
            assert bounds.shape == (size, size)

    def test_bounds_is_right(self):
        for size in range(1, 11):
            bounds = krippendorffs_alpha._calculation.get_summing_bounds(size)
            for c in range(size):
                for k in range(size):
                    if k > c:
                        assert bounds[c, k] == 1
                    else:
                        assert bounds[c, k] == 0


class TestMakeCoincidencesMatrixFromDataMatrix(object):
    def test_data_from_example_no_omit(self):
        expected_coincidences_matrix = numpy.array([
            [3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2, 0],
            [0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0, 0],
            [0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        ])

        actual_coincidences_matrix = krippendorffs_alpha._calculation.make_coincidences_matrix_from_data_matrix(
            DATA_MATRIX, omit_unpairable=False
        )
        assert expected_coincidences_matrix.shape == actual_coincidences_matrix.shape
        assert numpy.max(numpy.abs(expected_coincidences_matrix - actual_coincidences_matrix)) == pytest.approx(0), \
            f"expected:\n{expected_coincidences_matrix}\nactual:\n{actual_coincidences_matrix}"

    def test_data_from_example_omit_unpairable(self):
        expected_coincidences_matrix = numpy.array([
            [3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2],
            [0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0],
            [0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        ])

        actual_coincidences_matrix = krippendorffs_alpha._calculation.make_coincidences_matrix_from_data_matrix(
            DATA_MATRIX, omit_unpairable=True
        )
        assert expected_coincidences_matrix.shape == actual_coincidences_matrix.shape
        assert numpy.max(numpy.abs(expected_coincidences_matrix - actual_coincidences_matrix)) == pytest.approx(0), \
            f"expected:\n{expected_coincidences_matrix}\nactual:\n{actual_coincidences_matrix}"


class TestCalcAlpha(object):

    def test_e_nominal(self):
        coincidences_matrix = krippendorffs_alpha._calculation.make_coincidences_matrix_from_data_matrix(
            DATA_MATRIX, omit_unpairable=True
        )
        metric_tensor = krippendorffs_alpha.metrics.NominalMetric().get_metric_tensor(list(range(1, 6)))
        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(coincidences_matrix, metric_tensor)
        assert actual_alpha == pytest.approx(0.743, abs=0.001)

    def test_e_interval(self):
        coincidences_matrix = krippendorffs_alpha._calculation.make_coincidences_matrix_from_data_matrix(
            DATA_MATRIX,
            omit_unpairable=True)

        metric_tensor = krippendorffs_alpha.metrics.IntervalMetric().get_metric_tensor(list(range(1, 6)))
        actual_alpha = krippendorffs_alpha._calculation.calc_alpha(coincidences_matrix, metric_tensor)
        assert actual_alpha == pytest.approx(0.849, abs=0.001)
