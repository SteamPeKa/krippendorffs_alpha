# coding=utf-8
# Creation date: 19 окт. 2020
# Creation time: 10:22
# Creator: SteamPeKa

import warnings
from typing import TypeVar, Generic, Iterable, Union

import numpy

VT = TypeVar("VT")


class AbstractMetric(Generic[VT], object):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, value1: VT, value2: VT) -> float:
        raise NotImplementedError(
            "Abstract method of class AbstractMetric call from class {}".format(self.__class__.__name__)
        )

    def get_metric_tensor(self, possible_values: Iterable[VT], symmetric=True):
        if symmetric:
            return numpy.array(
                [[self(row_value, column_value) for column_value in possible_values] for row_value in possible_values]
            )
        else:
            return numpy.array([[self(row_value, column_value) if k > c else 0.0
                                 for k, column_value in enumerate(possible_values)]
                                for c, row_value in enumerate(possible_values)])


class NominalMetric(AbstractMetric[VT]):
    def __call__(self, value1: VT, value2: VT) -> float:
        return 0.0 if value1 == value2 else 1.0


class IntervalMetric(AbstractMetric[VT]):
    def __call__(self, value1: VT, value2: VT) -> float:
        return (value1 - value2) ** 2


class RatioMetric(AbstractMetric[VT]):
    def __call__(self, value1: VT, value2: VT) -> float:
        if value1 == value2:
            return 0.0
        return ((value1 - value2) / (value1 + value2)) ** 2


class CircularMetric(AbstractMetric[VT]):
    # noinspection PyUnusedLocal
    def __init__(self, possible_values_count: int = None, *args, **kwargs):
        super().__init__()
        if possible_values_count is None:
            raise ValueError("CircularMetric has to be initialised with number of possible intervals")
        elif not isinstance(possible_values_count, int):
            raise ValueError("Number of possible intervals have to be a natural number")
        elif possible_values_count < 1:
            raise ValueError("Number of possible intervals have to be a natural number")
        self.__U = possible_values_count

    def __call__(self, value1: VT, value2: VT) -> float:
        if value1 == value2:
            return 0.0
        difference_is_int = abs((value1 - value2) - int(value1 - value2)) < 1e-8
        if any([value1 > self.__U,
                value2 > self.__U,
                abs(value1 - value2) > self.__U,
                not difference_is_int]):
            warnings.warn(f"Input arguments for CircularMetric({value1},{value2})"
                          f" is not match the number of intervals of the circle {self.__U}")
        return numpy.square(numpy.sin(numpy.pi * ((value1 - value2) / self.__U)))


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
        elif "circular".startswith(metric):
            return CircularMetric(*args, **kwargs)
        else:
            raise ValueError("Unsupported metric name: {}".format(metric))
    else:
        raise ValueError("Unsupported metric-name value type: {} ({})".format(type(metric), str(metric)))
