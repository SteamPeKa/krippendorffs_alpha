# coding=utf-8
# Creation date: 27 окт. 2020
# Creation time: 18:43
# Creator: SteamPeKa

import typing

import numpy

from .data_converters import _PreparedData
from .metrics import AbstractMetric

"""
Protected member of package designed for encapsulate inconvenient  calculation calls and for unit-testing.
"""

__CREATED_SUMMING_BOUNDS = {}


def drop_summing_bounds_cache() -> None:
    """
    Clears cache of summing bounds arrays.
    """
    global __CREATED_SUMMING_BOUNDS
    __CREATED_SUMMING_BOUNDS.clear()


def _make_value_by_unit_matrix_from_data_matrix(data_matrix: numpy.ndarray,
                                                omit_unpairable: bool = True) -> numpy.ndarray:
    """
    :param data_matrix: Data matrix of shape (O, U, V). O -- number of unique observer,
                                                        U -- number of unique units,
                                                        V -- number of unique values
    :param omit_unpairable: If True the resulting matrix will have units that have more than one observer associated
                            with them only.
    :return: value_by_unit_matrix with unpairable units omitted if omit_unpairable is True.
    """
    assert len(data_matrix.shape) == 3
    norming_values = data_matrix.sum(axis=2)
    norming_values[norming_values == 0] = 1
    normed_data_matrix = data_matrix / norming_values[:, :, numpy.newaxis]
    raw_value_by_unit_matrix = normed_data_matrix.sum(axis=0).T
    if not omit_unpairable:
        return raw_value_by_unit_matrix

    observers_per_unit = numpy.zeros(data_matrix.shape[:2])
    observers_per_unit[(data_matrix != 0).any(axis=2)] = 1
    observers_per_unit = observers_per_unit.sum(axis=0)

    result = raw_value_by_unit_matrix[:, observers_per_unit > 1]
    return result


def _calculation_routine(value_by_unit_matrix: numpy.ndarray, bounded_distance_matrix: numpy.ndarray) -> float:
    """
    :param value_by_unit_matrix: Value-by-unit matrix of shape (V, U). V -- number of unique values,
                                                                       U -- number of unique units.
                                value_by_unit_matrix have to be a result of make_value_by_unit_matrix_from_data_matrix
                                function.
    :param bounded_distance_matrix: Upper diagonal matrix of shape (V, V) with squared distances between possible
                                    values. It is a point-wise multiplication of metric tensor and summing bounds
                                    matrix.
    :return: Krippendorff's alpha. A float value between -1.0 and 1.0.
    """
    assert len(value_by_unit_matrix.shape) == 2
    assert len(bounded_distance_matrix.shape) == 2
    assert bounded_distance_matrix.shape[0] == bounded_distance_matrix.shape[1]
    values_count = bounded_distance_matrix.shape[0]
    assert value_by_unit_matrix.shape[0] == values_count
    assert numpy.all(bounded_distance_matrix >= 0)
    # TODO assertion that bounded_distance_matrix is really bounded distance_matrix
    #     (everything lower main diagonal is zero).
    overlap = value_by_unit_matrix.sum(axis=0)
    assert numpy.all(overlap > 1), overlap
    answers_frequencies = value_by_unit_matrix.sum(axis=1)
    total_pairable_values = numpy.sum(answers_frequencies)
    assert total_pairable_values == numpy.sum(overlap)
    unit_norming = numpy.divide(1, overlap - 1)

    denominator = numpy.einsum("c,k,ck->", answers_frequencies, answers_frequencies, bounded_distance_matrix)
    numerator = numpy.einsum("u,cu,ku,ck->",
                             unit_norming, value_by_unit_matrix, value_by_unit_matrix, bounded_distance_matrix)

    result = 1.0 - (total_pairable_values - 1) * (numerator / denominator)
    assert -1.0 <= result <= 1.0
    return result


VT = typing.TypeVar("VT")


def _calc_alpha(W_tensor, bounded_distance_matrix):
    value_by_unit_matrix = _make_value_by_unit_matrix_from_data_matrix(W_tensor,
                                                                       omit_unpairable=True)
    return _calculation_routine(value_by_unit_matrix=value_by_unit_matrix,
                                bounded_distance_matrix=bounded_distance_matrix)


def calc_alpha(prepared_data: _PreparedData[typing.Any, typing.Any, VT],
               metric: typing.Union[None, AbstractMetric[VT], str]):
    bounded_distance_matrix = prepared_data.get_metric_tensor(metric, symmetric=False)
    value_by_unit_matrix = _make_value_by_unit_matrix_from_data_matrix(prepared_data.answers_tensor,
                                                                       omit_unpairable=True)
    return _calculation_routine(value_by_unit_matrix=value_by_unit_matrix,
                                bounded_distance_matrix=bounded_distance_matrix)


def _prepare_bootstrap_precomputes(prepared_data: _PreparedData[typing.Any, typing.Any, VT],
                                   metric: typing.Union[None, AbstractMetric[VT], str]):
    W_tensor = prepared_data.answers_tensor
    asymmetric_metric_tensor = prepared_data.get_metric_tensor(metric=metric, symmetric=False)

    assignment_matrix = W_tensor.sum(axis=2)
    assert assignment_matrix.shape == (prepared_data.observers_count, prepared_data.units_count)
    # noinspection SpellCheckingInspection
    full_cross_disagreement_tensor = numpy.einsum("imc,jsk,ck->ijms", W_tensor, W_tensor, asymmetric_metric_tensor)
    assert full_cross_disagreement_tensor.shape == (prepared_data.observers_count,
                                                    prepared_data.observers_count,
                                                    prepared_data.units_count,
                                                    prepared_data.units_count)
    return assignment_matrix, full_cross_disagreement_tensor


def _calc_alpha_by_precomputes(assignment_matrix: numpy.ndarray,
                               full_cross_disagreement_tensor: numpy.ndarray):
    observers_count, units_count = assignment_matrix.shape
    assert full_cross_disagreement_tensor.shape == (observers_count,
                                                    observers_count,
                                                    units_count,
                                                    units_count)
    units_overlaps = assignment_matrix.sum(axis=0)

    pairable_units_id = numpy.zeros(units_overlaps.shape)
    pairable_units_id[units_overlaps > 1] = 1

    total_pairable_answers = numpy.einsum("u,u->", pairable_units_id, units_overlaps)

    norming_values = numpy.zeros(units_overlaps.shape)
    norming_values[units_overlaps > 1] = 1.0 / (units_overlaps[units_overlaps > 1] - 1.0)

    # noinspection SpellCheckingInspection
    prepared_observed_disagreement = numpy.einsum("u,ijuu->", norming_values, full_cross_disagreement_tensor)
    prepared_expected_disagreement = numpy.einsum("m,s,ijms->",
                                                  pairable_units_id, pairable_units_id, full_cross_disagreement_tensor)

    return 1.0 - ((total_pairable_answers - 1) * (prepared_observed_disagreement / prepared_expected_disagreement))
