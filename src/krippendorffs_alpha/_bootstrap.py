# coding=utf-8
# Creation date: 04 нояб. 2020
# Creation time: 20:07
# Creator: SteamPeKa

from typing import Union

import numpy

from ._calculation import _PreparedData, _calc_alpha
from .metrics import AbstractMetric


def _precompute_tensors(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    bounded_metric_tensor = prepared_data.get_metric_tensor(metric=metric,
                                                            symmetric=False)

    W_tensor = prepared_data.answers_tensor.copy()
    assert W_tensor.shape == (prepared_data.observers_count,
                              prepared_data.units_count,
                              prepared_data.possible_values_count)

    Q_tensor = numpy.einsum("iuc,juk,ck->iju", W_tensor, W_tensor, bounded_metric_tensor)
    assert Q_tensor.shape == (prepared_data.observers_count,
                              prepared_data.observers_count,
                              prepared_data.units_count)

    P_tensor = W_tensor.sum(axis=2)
    return W_tensor, Q_tensor, P_tensor, bounded_metric_tensor


def _precompute_tensors_for_unitwise_resampling(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    W_tensor = prepared_data.answers_tensor
    metric_tensor = prepared_data.get_metric_tensor(metric, symmetric=False)

    value_by_unit = numpy.einsum("ouv->uv", W_tensor)

    overlaps = numpy.einsum("uv->u", value_by_unit)
    not_pairable_units = overlaps <= 1
    overlaps[not_pairable_units] = 0
    value_by_unit[not_pairable_units, :] = 0

    norming_values = 1.0 / (overlaps - 1.0)
    norming_values[not_pairable_units] = 0

    numerator_outer_sums = norming_values * numpy.einsum("iuc,juk,ck->u", W_tensor, W_tensor, metric_tensor)
    D_matrix = numpy.einsum("nc,mk,ck->nm", value_by_unit, value_by_unit, metric_tensor)
    return overlaps, numerator_outer_sums, D_matrix


def _prepare_jackknife_indexing(total):
    result = numpy.tile(numpy.array(range(total), dtype=numpy.int64), (total, 1))
    numpy.fill_diagonal(result, 0)
    result = result[:, 1:]
    return result


def _ram_hungry_observerwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    if prepared_data.observers_count <= 2:
        raise ValueError("Not enough observers for valid jackknifing results")
    W_tensor, Q_tensor, P_tensor, bounded_metric_tensor = _precompute_tensors(prepared_data=prepared_data,
                                                                              metric=metric)
    observers_choices = _prepare_jackknife_indexing(prepared_data.observers_count)

    norming_values = (P_tensor[observers_choices, :]).sum(axis=1)
    unpairable_units_masks = norming_values <= 1
    norming_values[unpairable_units_masks] = 0.0
    norming_values = 1.0 / (norming_values - 1.0)
    norming_values[unpairable_units_masks] = 0.0

    Q_col_sums = Q_tensor.sum(axis=1)
    numerators = numpy.einsum("bu,bu->b",
                              (
                                      (numpy.tile(Q_col_sums.sum(axis=0), (prepared_data.observers_count, 1)))
                                      - Q_col_sums  # I have no idea why Q_tensor is not symmetric
                                      - Q_tensor.sum(axis=0)  # but Q[i,j,:] != Q[j,i,:].
                                      + numpy.einsum("iiu->iu", Q_tensor)
                              ),
                              norming_values)

    frequencies = (W_tensor[observers_choices, :, :]).sum(axis=1)
    frequencies[unpairable_units_masks, :] = 0
    frequencies = frequencies.sum(axis=1)

    total_pairable_for_resample = frequencies.sum(axis=1)

    denominators = numpy.einsum("bc,bk,ck->b", frequencies, frequencies, bounded_metric_tensor)

    result = 1.0 - ((total_pairable_for_resample - 1.0) * (numerators / denominators))
    return numpy.array(result)


def _time_hungry_observerwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    if prepared_data.observers_count <= 2:
        raise ValueError("Not enough observers for valid jackknifing results")
    observers_choices = _prepare_jackknife_indexing(prepared_data.observers_count)
    bounded_distance_matrix = prepared_data.get_metric_tensor(metric, symmetric=False)
    W_tensor = prepared_data.answers_tensor
    result = []
    for indexing in observers_choices:
        result.append(_calc_alpha(W_tensor=W_tensor[indexing, :, :],
                                  bounded_distance_matrix=bounded_distance_matrix))

    return numpy.array(result)


def observerwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric],
                           strategy="ram-hungry"):
    if strategy not in {"ram-hungry", "time-hungry"}:
        raise ValueError(f"Unsupported strategy value: {strategy}")
    if strategy == "ram-hungry":
        return _ram_hungry_observerwise_jackknife(prepared_data=prepared_data,
                                                  metric=metric)
    elif strategy == "time-hungry":
        return _time_hungry_observerwise_jackknife(prepared_data=prepared_data,
                                                   metric=metric)
    else:
        assert False


def unitwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    overlaps, numerator_outer_sums, D_matrix = _precompute_tensors_for_unitwise_resampling(prepared_data=prepared_data,
                                                                                           metric=metric)
    D_row_sums = D_matrix.sum(axis=0)
    D_col_sums = D_matrix.sum(axis=1)
    S_value = D_row_sums.sum(axis=0)
    D_diagonal = numpy.einsum("ii->i", D_matrix)

    totals_for_batches = overlaps.sum() - overlaps
    numerators_for_batches = numerator_outer_sums.sum() - numerator_outer_sums
    denominator_for_batches = S_value - D_row_sums - D_col_sums + D_diagonal

    result = 1 - ((totals_for_batches - 1) * (numerators_for_batches / denominator_for_batches))
    return result
