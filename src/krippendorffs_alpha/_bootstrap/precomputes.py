# coding=utf-8
# Creation date: 17 нояб. 2020
# Creation time: 11:58
# Creator: SteamPeKa
import numpy


def _prepare_general_resampling_precomputes(weights_encoded_answers_data_matrix: numpy.ndarray,
                                            metric_tensor: numpy.ndarray):
    observers_count, units_count, values_count = weights_encoded_answers_data_matrix.shape
    assert metric_tensor.shape == (values_count, values_count)
    assignment_matrix = weights_encoded_answers_data_matrix.sum(axis=2)
    assert numpy.min(assignment_matrix) >= 0
    assert numpy.max(assignment_matrix) <= 1
    # noinspection SpellCheckingInspection
    full_cross_disagreement_tensor = numpy.einsum("imc,jsk,ck->ijms",
                                                  weights_encoded_answers_data_matrix,
                                                  weights_encoded_answers_data_matrix,
                                                  metric_tensor)
    return full_cross_disagreement_tensor, assignment_matrix


def _prepare_unitwise_only_resampling_precomputes(weights_encoded_answers_data_matrix: numpy.ndarray,
                                                  metric_tensor: numpy.ndarray):
    cross_unit_disagreement = numpy.einsum("imc,jsk,ck->ms",
                                           weights_encoded_answers_data_matrix,
                                           weights_encoded_answers_data_matrix,
                                           metric_tensor)
    units_overlaps = numpy.einsum("ouv->u",
                                  weights_encoded_answers_data_matrix)

    pairable_units_bool_mask = units_overlaps > 1
    unpairable_units_bool_mask = numpy.invert(pairable_units_bool_mask)
    pairable_units_num_mask = numpy.zeros(shape=pairable_units_bool_mask.shape)
    pairable_units_num_mask[pairable_units_bool_mask] = 1.0

    pairable_overlaps = numpy.zeros(shape=units_overlaps.shape)
    pairable_overlaps[unpairable_units_bool_mask] = 0

    norming_values = numpy.zeros(shape=units_overlaps.shape)
    norming_values[pairable_units_bool_mask] = 1.0 / (units_overlaps[pairable_units_bool_mask] - 1)

    pairable_cross_unit_disagreement = numpy.einsum("n,m,nm->nm",
                                                    pairable_units_num_mask,
                                                    pairable_units_num_mask,
                                                    cross_unit_disagreement)

    return pairable_cross_unit_disagreement, pairable_overlaps, norming_values
