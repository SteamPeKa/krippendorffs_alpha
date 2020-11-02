# coding=utf-8
# Creation date: 19 окт. 2020
# Creation time: 10:22
# Creator: SteamPeKa


from typing import TypeVar, Generic, Iterable, Union

import numpy

T = TypeVar("T")


class AbstractMetric(Generic[T], object):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, value1: T, value2: T) -> float:
        raise NotImplementedError(
            "Abstract method of class AbstractMetric call from class {}".format(self.__class__.__name__)
        )

    def get_metric_tensor(self, possible_values: Iterable[T], symmetric=True):
        if symmetric:
            return numpy.array(
                [[self(row_value, column_value) for column_value in possible_values] for row_value in possible_values]
            )
        else:
            return numpy.array([[self(row_value, column_value) if k > c else 0.0
                                 for k, column_value in enumerate(possible_values)]
                                for c, row_value in enumerate(possible_values)])


class NominalMetric(AbstractMetric):
    def __call__(self, value1: T, value2: T) -> float:
        return 0.0 if value1 == value2 else 1.0


class IntervalMetric(AbstractMetric):
    def __call__(self, value1: T, value2: T) -> float:
        return (value1 - value2) ** 2


class RatioMetric(AbstractMetric):
    def __call__(self, value1: T, value2: T) -> float:
        if value1 == value2:
            return 0.0
        return ((value1 - value2) / (value1 + value2)) ** 2


def get_metric(metric: Union[None, AbstractMetric, str], *args, **kwargs) -> AbstractMetric:
    if metric is None:
        return NominalMetric(*args, **kwargs)
    elif isinstance(metric, AbstractMetric):
        return metric
    elif isinstance(metric, str):
        if len(metric) == 0:
            raise ValueError("Empty metric-name string!")
        if "nominal".startswith(metric):
            return NominalMetric(*args, **kwargs)
        elif "interval".startswith(metric):
            return IntervalMetric(*args, **kwargs)
        elif "ratio".startswith(metric):
            return RatioMetric(*args, **kwargs)
        else:
            raise ValueError("Unsupported metric name: {}".format(metric))
    else:
        raise ValueError("Unsupported metric-name value type: {} ({})".format(type(metric), str(metric)))
