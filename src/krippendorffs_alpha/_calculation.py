# coding=utf-8
# Creation date: 27 окт. 2020
# Creation time: 18:43
# Creator: SteamPeKa

import numpy

"""
Protected member of package designed for encapsulate inconvenient  calculation calls and for unit-testing.
"""

__CREATED_SUMMING_BOUNDS = {}


def get_summing_bounds(size: int) -> numpy.ndarray:
    """
    Function producing necessary immutable "summing bounds" matrices. Encapsulates reusing.

    :param size: positive integer defining shape of resulting array
    :return: Upper triangular matrix of ones with zeroed main diagonal and all diagonals bellow
    """
    if size <= 0:
        raise ValueError(
            f"Can't create squared matrix of shape ({size},{size}). Param size has to be a natural number."
        )
    global __CREATED_SUMMING_BOUNDS
    if size not in __CREATED_SUMMING_BOUNDS:
        result = numpy.triu(numpy.ones(shape=(size, size)), 1)
        result.flags.writeable = False
        __CREATED_SUMMING_BOUNDS[size] = result
    return __CREATED_SUMMING_BOUNDS[size]


def make_coincidences_matrix_from_data_matrix(data_matrix: numpy.ndarray,
                                              omit_unpairable: bool = True) -> numpy.ndarray:
    """
    :param data_matrix: Data matrix of shape (O, U, V). O -- number of unique observer,
                                                        U -- number of unique units,
                                                        V -- number of unique values
    :param omit_unpairable: If True the resulting matrix will have units that have more than one observer associated
                            with them only.
    :return: coincidences_matrix with unpairable units omitted if omit_unpairable is True.
    """
    assert len(data_matrix.shape) == 3
    norming_values = data_matrix.sum(axis=2)
    norming_values[norming_values == 0] = 1
    normed_data_matrix = data_matrix / norming_values[:, :, numpy.newaxis]
    raw_coincidences_matrix = normed_data_matrix.sum(axis=0).T
    if not omit_unpairable:
        return raw_coincidences_matrix

    observers_per_unit = numpy.zeros(data_matrix.shape[:2])
    observers_per_unit[(data_matrix != 0).any(axis=2)] = 1
    observers_per_unit = observers_per_unit.sum(axis=0)

    result = raw_coincidences_matrix[:, observers_per_unit > 1]
    return result


def calc_alpha(coincidences_matrix: numpy.ndarray, distance_matrix: numpy.ndarray) -> float:
    """
    :param coincidences_matrix: Coincidences matrix of shape (V, U). V -- number of unique values,
                                                                     U -- number of unique units.
                                coincidences_matrix have to be a result of make_coincidences_matrix_from_data_matrix
                                function.
    :param distance_matrix: Matrix of shape (V, V) with squared distances between possible values.
    :return: Krippendorff's alpha. A float value between -1.0 and 1.0.
    """
    assert len(coincidences_matrix.shape) == 2
    assert len(distance_matrix.shape) == 2
    assert coincidences_matrix.shape[0] == distance_matrix.shape[0] == distance_matrix.shape[1]
    assert numpy.all(distance_matrix >= 0)
    values_count = distance_matrix.shape[1]
    overlap = coincidences_matrix.sum(axis=0)
    assert numpy.all(overlap > 1), overlap
    answers_frequencies = coincidences_matrix.sum(axis=1)
    total_pairable_values = numpy.sum(answers_frequencies)
    assert total_pairable_values == numpy.sum(overlap)
    inner_summing_bounds = get_summing_bounds(values_count)
    unit_norming = numpy.divide(1, overlap - 1)

    denominator = numpy.einsum("ck,c,k,ck->",
                               inner_summing_bounds, answers_frequencies, answers_frequencies, distance_matrix)
    numerator = numpy.einsum("u,ck,cu,ku,ck->",
                             unit_norming, inner_summing_bounds, coincidences_matrix, coincidences_matrix,
                             distance_matrix)

    result = 1.0 - (total_pairable_values - 1) * (numerator / denominator)
    assert -1.0 <= result <= 1.0
    return result
