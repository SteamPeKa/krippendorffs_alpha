# coding=utf-8
# Creation date: 04 нояб. 2020
# Creation time: 20:07
# Creator: SteamPeKa

from typing import Union

import numpy

from ._calculation import _PreparedData
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


def prepare_jackknife_indexing(total):
    result = numpy.tile(numpy.array(range(total)), (total, 1))
    numpy.fill_diagonal(result, 0)
    result = result[:, 1:]
    return result


def observerwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    if prepared_data.observers_count <= 2:
        raise ValueError("Not enough observers for valid jackknifing results")
    W_tensor, Q_tensor, P_tensor, bounded_metric_tensor = _precompute_tensors(prepared_data=prepared_data,
                                                                              metric=metric)
    observers_choices = prepare_jackknife_indexing(prepared_data.observers_count)

    norming_values = (P_tensor[observers_choices, :]).sum(axis=1)
    unpairable_units_masks = norming_values <= 1
    norming_values[unpairable_units_masks] = 0.0
    norming_values = 1.0 / (norming_values - 1.0)
    norming_values[unpairable_units_masks] = 0.0

    numerators = numpy.einsum("bu,bu->b",
                              (
                                      (numpy.tile(Q_tensor.sum(axis=(0, 1)), (prepared_data.observers_count, 1)))
                                      - Q_tensor.sum(axis=1)  # I have no idea why Q_tensor is not symmetric
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
