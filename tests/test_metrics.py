# coding=utf-8
# Creation date: 19 окт. 2020
# Creation time: 10:13
# Creator: SteamPeKa

import pytest
import krippendorffs_alpha
import random
import numpy

random.seed(42)


class TestGetMetrics(object):
    def test_get_by_empty_string(self):
        with pytest.raises(ValueError) as exception_info:
            krippendorffs_alpha.metrics.get_metric("")
        assert exception_info.type is ValueError

    def test_get_metric_interval(self):
        metric_name = "interval"
        for i in range(len(metric_name)):
            metric_object = krippendorffs_alpha.metrics.get_metric(metric_name[:i + 1])
            assert isinstance(metric_object, krippendorffs_alpha.metrics.AbstractMetric)
            assert isinstance(metric_object, krippendorffs_alpha.metrics.IntervalMetric)

    def test_get_metric_nominal(self):
        metric_name = "nominal"
        for i in range(len(metric_name)):
            metric_object = krippendorffs_alpha.metrics.get_metric(metric_name[:i + 1])
            assert isinstance(metric_object, krippendorffs_alpha.metrics.AbstractMetric)
            assert isinstance(metric_object, krippendorffs_alpha.metrics.NominalMetric)

    def test_get_metric_ratio(self):
        metric_name = "ratio"
        for i in range(len(metric_name)):
            metric_object = krippendorffs_alpha.metrics.get_metric(metric_name[:i + 1])
            assert isinstance(metric_object, krippendorffs_alpha.metrics.AbstractMetric)
            assert isinstance(metric_object, krippendorffs_alpha.metrics.RatioMetric)

    def test_get_metric_by_none(self):
        metric_object = krippendorffs_alpha.metrics.get_metric(None)
        assert isinstance(metric_object, krippendorffs_alpha.metrics.AbstractMetric)
        assert isinstance(metric_object, krippendorffs_alpha.metrics.NominalMetric)


class TestNominalMetric(object):
    def test_nominal_metric_is_a_metric(self):
        metric_object = krippendorffs_alpha.metrics.NominalMetric()
        cases_count = 10
        for _ in range(cases_count):
            v = random.expovariate(lambd=10)
            assert metric_object(v, v) == pytest.approx(0)
        for _ in range(cases_count):
            v1 = random.expovariate(lambd=10)
            v2 = random.expovariate(lambd=10)
            assert metric_object(v1, v2) == pytest.approx(metric_object(v2, v1))
            assert metric_object(v1, v2) >= 0

    def test_values(self):
        metric_object = krippendorffs_alpha.metrics.NominalMetric()
        test_cases = [
            ((0, 0), 0),
            ((0, 1), 1),
            ((1, 0), 1),
            ((1, 1), 0),
        ]
        for (v1, v2), true_result in test_cases:
            assert metric_object(v1, v2) == pytest.approx(true_result)

    def test_get_metric_matrix(self):
        metric = krippendorffs_alpha.metrics.NominalMetric()
        expected_matrix = numpy.array([[0, 1, 1, 1, 1],
                                       [1, 0, 1, 1, 1],
                                       [1, 1, 0, 1, 1],
                                       [1, 1, 1, 0, 1],
                                       [1, 1, 1, 1, 0]])
        result_matrix = metric.get_metric_tensor([0, 1, 2, 3, 4])
        assert numpy.max(numpy.abs(expected_matrix - result_matrix)) == pytest.approx(0)


class TestIntervalMetric(object):

    def test_interval_metric_is_a_metric(self):
        metric_object = krippendorffs_alpha.metrics.IntervalMetric()
        cases_count = 10
        for _ in range(cases_count):
            v = random.expovariate(lambd=10)
            assert metric_object(v, v) == pytest.approx(0)
        for _ in range(cases_count):
            v1 = random.expovariate(lambd=10)
            v2 = random.expovariate(lambd=10)
            assert metric_object(v1, v2) == pytest.approx(metric_object(v2, v1))
            assert metric_object(v1, v2) > 0

    def test_values(self):
        metric_object = krippendorffs_alpha.metrics.IntervalMetric()
        test_cases = [
            ((10, 20), 100),
            ((7, 3), 16),
            ((0, -1), 1),
            ((-5, -7), 4),
            ((0, 0), 0),
            ((-43, -1), 42 ** 2),
        ]
        for (v1, v2), true_result in test_cases:
            assert metric_object(v1, v2) == pytest.approx(true_result)

    def test_get_metric_matrix(self):
        metric = krippendorffs_alpha.metrics.IntervalMetric()
        expected_matrix = numpy.array([[0, 1, 4, 9, 16],
                                       [1, 0, 1, 4, 9],
                                       [4, 1, 0, 1, 4],
                                       [9, 4, 1, 0, 1],
                                       [16, 9, 4, 1, 0]])
        result_matrix = metric.get_metric_tensor([0, 1, 2, 3, 4])
        assert numpy.max(numpy.abs(expected_matrix - result_matrix)) == pytest.approx(0)


class TestRatioMetric(object):

    def test_ratio_metric_is_a_metric(self):
        metric_object = krippendorffs_alpha.metrics.RatioMetric()
        cases_count = 10
        for _ in range(cases_count):
            v = random.expovariate(lambd=10)
            assert metric_object(v, v) == pytest.approx(0)
        for _ in range(cases_count):
            v1 = random.expovariate(lambd=10)
            v2 = random.expovariate(lambd=10)
            assert metric_object(v1, v2) == pytest.approx(metric_object(v2, v1))
            assert metric_object(v1, v2) > 0

    def test_values(self):
        metric_object = krippendorffs_alpha.metrics.RatioMetric()
        test_cases = [
            ((10, 20), 1 / 9),
            ((7, 3), 16 / 100),
            ((0, -1), 1),
            ((-5, -7), 4 / 144),
            ((0, 0), 0),
        ]
        for (v1, v2), true_result in test_cases:
            assert metric_object(v1, v2) == pytest.approx(true_result)

    def test_get_metric_matrix(self):
        metric = krippendorffs_alpha.metrics.RatioMetric()
        expected_matrix = numpy.array([[0, 1, 1, 1, 1],
                                       [1, 0, 1 / 9, 1 / 4, 9 / 25],
                                       [1, 1 / 9, 0, 1 / 25, 4 / 36],
                                       [1, 1 / 4, 1 / 25, 0, 1 / 49],
                                       [1, 9 / 25, 4 / 36, 1 / 49, 0]])
        result_matrix = metric.get_metric_tensor([0, 1, 2, 3, 4])
        assert numpy.max(numpy.abs(expected_matrix - result_matrix)) == pytest.approx(0), (
                expected_matrix - result_matrix)
