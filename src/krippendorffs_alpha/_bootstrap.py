# coding=utf-8
# Creation date: 04 нояб. 2020
# Creation time: 20:07
# Creator: SteamPeKa

from typing import Union

import numpy

from ._calculation import _PreparedData, _calc_alpha, _prepare_bootstrap_precomputes
from .metrics import AbstractMetric


def _precompute_tensors_for_unitwise_resampling(prepared_data: _PreparedData,
                                                metric: Union[None, str, AbstractMetric]):
    assignment_matrix, full_cross_disagreement_tensor = _prepare_bootstrap_precomputes(prepared_data=prepared_data,
                                                                                       metric=metric)
    # noinspection SpellCheckingInspection
    cross_unit_disagreement = numpy.einsum("ijm"
                                           "s->ms", full_cross_disagreement_tensor)
    units_overlaps = assignment_matrix.sum(axis=0)

    pairable_units_num_mask = numpy.zeros(units_overlaps.shape)
    pairable_units_num_mask[units_overlaps > 1] = 1.0

    pairable_overlaps = numpy.zeros(units_overlaps.shape)
    pairable_overlaps[units_overlaps > 1] = units_overlaps[units_overlaps > 1]

    norming_values = numpy.zeros(units_overlaps.shape)
    norming_values[units_overlaps > 1] = 1.0 / (units_overlaps[units_overlaps > 1] - 1)

    pairable_cross_unit_disagreement = numpy.einsum("n,m,nm->nm",
                                                    pairable_units_num_mask,
                                                    pairable_units_num_mask,
                                                    cross_unit_disagreement)

    return pairable_cross_unit_disagreement, pairable_overlaps, norming_values


def _prepare_jackknife_indexing(total):
    result = numpy.tile(numpy.array(range(total), dtype=numpy.int64), (total, 1))
    numpy.fill_diagonal(result, 0)
    result = result[:, 1:]
    return result


def _ram_hungry_observerwise_jackknife(prepared_data: _PreparedData, metric: Union[None, str, AbstractMetric]):
    if prepared_data.observers_count <= 2:
        raise ValueError("Not enough observers for valid jackknifing results")
    assignment_matrix, full_cross_disagreement_tensor = _prepare_bootstrap_precomputes(prepared_data=prepared_data,
                                                                                       metric=metric)

    observers_masks = numpy.ones((prepared_data.observers_count, prepared_data.observers_count))
    numpy.fill_diagonal(observers_masks, 0)

    overlaps = numpy.einsum("bo,ou->bu", observers_masks, assignment_matrix)

    pairable_unit_id_for_batches = numpy.zeros(overlaps.shape)
    pairable_unit_id_for_batches[overlaps > 1] = 1

    norming_values = numpy.zeros(overlaps.shape)
    norming_values[overlaps > 1] = 1.0 / (overlaps[overlaps > 1] - 1)

    full_cross_disagreement_tensor_for_batches = numpy.einsum("bi,bj,ijms->bijms",
                                                              observers_masks,
                                                              observers_masks,
                                                              full_cross_disagreement_tensor)
    # noinspection SpellCheckingInspection
    prepared_observed_disagreements = numpy.einsum("bu,bijuu->b",
                                                   norming_values,
                                                   full_cross_disagreement_tensor_for_batches)
    # noinspection SpellCheckingInspection
    prepared_expected_disagreement = numpy.einsum("bm,bs,bijms->b",
                                                  pairable_unit_id_for_batches,
                                                  pairable_unit_id_for_batches,
                                                  full_cross_disagreement_tensor_for_batches)

    pairable_answers_totals = numpy.einsum("bu,bu->b",
                                           pairable_unit_id_for_batches,
                                           overlaps)

    result = 1.0 - ((pairable_answers_totals - 1.0) * (prepared_observed_disagreements /
                                                       prepared_expected_disagreement))
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
    (pairable_cross_unit_disagreement,
     pairable_overlaps,
     norming_values) = _precompute_tensors_for_unitwise_resampling(prepared_data=prepared_data, metric=metric)

    row_disagreements = pairable_cross_unit_disagreement.sum(axis=0)
    column_disagreements = pairable_cross_unit_disagreement.sum(axis=1)
    overall_disagreement = pairable_cross_unit_disagreement.sum()
    units_inner_disagreement = numpy.einsum("ii->i", pairable_cross_unit_disagreement)

    expected_disagreements_for_batches = (overall_disagreement - row_disagreements - column_disagreements +
                                          units_inner_disagreement)

    totals_for_batches = pairable_overlaps.sum() - pairable_overlaps

    observed_units_disagreements = norming_values * units_inner_disagreement

    observed_disagreements_for_batches = observed_units_disagreements.sum() - observed_units_disagreements

    result = 1 - ((totals_for_batches - 1) * (observed_disagreements_for_batches / expected_disagreements_for_batches))
    return result
