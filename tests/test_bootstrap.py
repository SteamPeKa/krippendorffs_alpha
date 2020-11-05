# coding=utf-8
# Creation date: 04 нояб. 2020
# Creation time: 21:24
# Creator: SteamPeKa
import csv
import os

import numpy

import krippendorffs_alpha
import testing_utils


class TestObserverwiseJackknife(object):

    def test_values_nominal(self):
        metric = "nominal"
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = [row for row in csv.reader(f, delimiter="\t")]
        input_table = input_table[1:]
        input_table = [[int(val) if val.strip() != "NULL" else None for val in row[1:]] for row in input_table]

        users_count = len(input_table)
        expected_outed_alphas = []
        for user_index in range(users_count):
            jackknife_table = input_table[:user_index] + input_table[user_index + 1:]
            prepared_jackknife_table = krippendorffs_alpha.data_converters.from_list_of_lists(
                jackknife_table,
                header=False,
                row_legend=False,
                upper_level="observer",
                value_constructor=None,
                possible_values=[1, 2, 3, 4, 5])
            outer_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_jackknife_table, metric=metric)
            expected_outed_alphas.append(outer_alpha)

        prepared_input_table = krippendorffs_alpha.data_converters.from_list_of_lists(input_table,
                                                                                      header=False,
                                                                                      row_legend=False,
                                                                                      upper_level="observer",
                                                                                      value_constructor=None,
                                                                                      possible_values=[1, 2, 3, 4, 5])
        actual_outer_alphas = krippendorffs_alpha._bootstrap.observerwise_jackknife(prepared_data=prepared_input_table,
                                                                                    metric=metric)
        expected_outed_alphas = numpy.array(expected_outed_alphas)
        assert len(expected_outed_alphas) == len(actual_outer_alphas), (len(expected_outed_alphas),
                                                                        len(actual_outer_alphas),
                                                                        actual_outer_alphas)
        testing_utils.assert_equal_tensors(expected_outed_alphas, actual_outer_alphas)

    def test_values_interval(self):
        metric = "interval"
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = [row for row in csv.reader(f, delimiter="\t")]
        input_table = input_table[1:]
        input_table = [[int(val) if val.strip() != "NULL" else None for val in row[1:]] for row in input_table]

        users_count = len(input_table)
        expected_outed_alphas = []
        for user_index in range(users_count):
            jackknife_table = input_table[:user_index] + input_table[user_index + 1:]
            prepared_jackknife_table = krippendorffs_alpha.data_converters.from_list_of_lists(
                jackknife_table,
                header=False,
                row_legend=False,
                upper_level="observer",
                value_constructor=None,
                possible_values=[1, 2, 3, 4, 5])
            outer_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_jackknife_table, metric=metric)
            expected_outed_alphas.append(outer_alpha)

        prepared_input_table = krippendorffs_alpha.data_converters.from_list_of_lists(input_table,
                                                                                      header=False,
                                                                                      row_legend=False,
                                                                                      upper_level="observer",
                                                                                      value_constructor=None,
                                                                                      possible_values=[1, 2, 3, 4, 5])
        actual_outer_alphas = krippendorffs_alpha._bootstrap.observerwise_jackknife(prepared_data=prepared_input_table,
                                                                                    metric=metric)
        expected_outed_alphas = numpy.array(expected_outed_alphas)
        assert len(expected_outed_alphas) == len(actual_outer_alphas), (len(expected_outed_alphas),
                                                                        len(actual_outer_alphas),
                                                                        actual_outer_alphas)
        testing_utils.assert_equal_tensors(expected_outed_alphas, actual_outer_alphas)

    def test_values_ratio(self):
        metric = "ratio"
        with open(os.path.join("tests", "example_E_data.tsv"), "r") as f:
            input_table = [row for row in csv.reader(f, delimiter="\t")]
        input_table = input_table[1:]
        input_table = [[int(val) if val.strip() != "NULL" else None for val in row[1:]] for row in input_table]

        users_count = len(input_table)
        expected_outed_alphas = []
        for user_index in range(users_count):
            jackknife_table = input_table[:user_index] + input_table[user_index + 1:]
            prepared_jackknife_table = krippendorffs_alpha.data_converters.from_list_of_lists(
                jackknife_table,
                header=False,
                row_legend=False,
                upper_level="observer",
                value_constructor=None,
                possible_values=[1, 2, 3, 4, 5])
            outer_alpha = krippendorffs_alpha._calculation.calc_alpha(prepared_jackknife_table, metric=metric)
            expected_outed_alphas.append(outer_alpha)

        prepared_input_table = krippendorffs_alpha.data_converters.from_list_of_lists(input_table,
                                                                                      header=False,
                                                                                      row_legend=False,
                                                                                      upper_level="observer",
                                                                                      value_constructor=None,
                                                                                      possible_values=[1, 2, 3, 4, 5])
        actual_outer_alphas = krippendorffs_alpha._bootstrap.observerwise_jackknife(prepared_data=prepared_input_table,
                                                                                    metric=metric)
        expected_outed_alphas = numpy.array(expected_outed_alphas)
        assert len(expected_outed_alphas) == len(actual_outer_alphas), (len(expected_outed_alphas),
                                                                        len(actual_outer_alphas),
                                                                        actual_outer_alphas)
        testing_utils.assert_equal_tensors(expected_outed_alphas, actual_outer_alphas)
