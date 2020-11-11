# coding=utf-8
# Creation date: 28 окт. 2020
# Creation time: 15:30
# Creator: SteamPeKa

from typing import Iterable, Union, Callable, List, Generic, Dict, TypeVar, Tuple

import numpy

from . import metrics

VT = TypeVar("VT")  # Answer Value type
UT = TypeVar("UT")  # Unit-key type
OT = TypeVar("OT")  # Observer-key type
RT = TypeVar("RT")  # Raw-content value type


class _PreparedData(Generic[OT, UT, VT], object):
    __possible_values: Tuple[VT]

    def __init__(self, answers_tensor: numpy.ndarray, possible_values: Iterable[VT],
                 units_names: Union[None, Iterable[UT]] = None, observers_names: Union[None, Iterable[OT]] = None):
        assert len(answers_tensor.shape) == 3, answers_tensor.shape  # (O, U, V)

        norming_values = answers_tensor.sum(axis=2)
        norming_values[norming_values == 0] = 1
        norming_values = 1.0 / norming_values

        self.__answers_tensor = numpy.einsum("ouv,ou->ouv", answers_tensor, norming_values)

        self.__answers_tensor.flags.writeable = False
        possible_values = tuple(possible_values)
        assert len(possible_values) == answers_tensor.shape[2]
        self.__possible_values = possible_values
        if units_names is not None:
            units_names = tuple(units_names)
            if self.__answers_tensor.shape[1] != len(units_names):
                raise ValueError("{}, {}".format(units_names, self.__answers_tensor.shape))
            self.__units_names = units_names
        else:
            self.__units_names = None
        if observers_names is not None:
            observers_names = tuple(observers_names)
            if self.__answers_tensor.shape[0] != len(observers_names):
                raise ValueError("{}, {}".format(observers_names, self.__answers_tensor.shape))

            self.__observers_names = observers_names
        else:
            self.__observers_names = None

    @property
    def answers_tensor(self) -> numpy.ndarray:
        return self.__answers_tensor

    @property
    def possible_values(self) -> Tuple[VT]:
        return self.__possible_values

    @property
    def observers_count(self) -> int:
        return self.answers_tensor.shape[0]

    @property
    def units_count(self) -> int:
        return self.answers_tensor.shape[1]

    @property
    def possible_values_count(self):
        return len(self.possible_values)

    @property
    def observers_names(self) -> Union[Tuple[int], Tuple[OT]]:
        if self.__observers_names is None:
            return tuple(i for i in range(self.observers_count))
        else:
            assert self.observers_count == len(self.__observers_names)
            return self.__observers_names

    @property
    def units_names(self) -> Union[Tuple[int], Tuple[UT]]:
        if self.__units_names is None:
            return tuple(i for i in range(self.units_count))
        else:
            assert self.units_count == len(self.__units_names)
            return self.__units_names

    def get_metric_tensor(self, metric="nominal", symmetric=True):
        return metrics.get_metric(metric).get_metric_tensor(self.possible_values, symmetric=symmetric)


def init_possible_values(possible_values: Union[None, Iterable[VT]]) -> Tuple[Union[List[VT], Tuple[VT]], bool]:
    if possible_values is None:
        return [], True
    else:
        _checking_possible_values = []
        for pos, v in enumerate(possible_values):
            if v in _checking_possible_values:
                raise ValueError(f"Passed possible_values contains copies of same element.\n"
                                 f"Elements with index {pos} and {_checking_possible_values.index(v)} are equal\n"
                                 f"Passed possible_values: {possible_values}")
            _checking_possible_values.append(v)
        return tuple(_checking_possible_values), False


def from_dict_of_dicts(input_table: Union[Dict[Union[OT, UT], Dict[Union[OT, UT], Union[RT, VT, None]]],
                                          Dict[Union[OT, UT], Dict[Union[OT, UT], List[Union[RT, VT, None]]]]],
                       upper_level: str = "observer",
                       value_constructor: Union[None, Callable[[RT], VT]] = None,
                       possible_values: Union[None, Iterable[VT]] = None):
    """
    :param input_table:
    :param upper_level: {"observer","unit"}
    :param value_constructor:
    :param possible_values:
    :return:
    """

    if upper_level == "observer":
        observers_names = tuple(input_table.keys())
        units_names = set()
        for observer_id in observers_names:
            units_names.update(input_table[observer_id].keys())
        units_names = tuple(units_names)
    elif upper_level == "unit":
        units_names = tuple(input_table.keys())
        observers_names = set()
        for unit_id in units_names:
            observers_names.update(input_table[unit_id].keys())
        observers_names = tuple(observers_names)
    else:
        raise ValueError(f"Unsupported upper_level type: {upper_level}")

    def _prepare_value(_raw_value) -> Union[None, List[VT]]:
        if _raw_value is None:
            return []
        if isinstance(_raw_value, list):
            _raw_answers_iterator = _raw_value
        else:
            _raw_answers_iterator = [_raw_value]
        _result = []
        for _raw_answer in _raw_answers_iterator:
            if _raw_answer is None:
                continue
            if value_constructor is None:
                _result.append(_raw_answer)
            else:
                _result.append(value_constructor(_raw_answer))
        return _result

    possible_values, track_value = init_possible_values(possible_values)
    if track_value:
        for lower_key_to_raw_answer in input_table.values():
            for raw_value in lower_key_to_raw_answer.values():
                for answer in _prepare_value(raw_value):
                    if answer not in possible_values:
                        possible_values.append(answer)
    possible_values = tuple(possible_values)

    answers_tensor = numpy.zeros(shape=(len(observers_names), len(units_names), len(possible_values)))
    for upper_key, lower_key_to_raw_answer in input_table.items():
        for lower_key, raw_answer in lower_key_to_raw_answer.items():
            if upper_level == "observer":
                observer_id = upper_key
                unit_id = lower_key
            elif upper_level == "unit":
                unit_id = upper_key
                observer_id = lower_key
            else:
                raise ValueError(f"Unsupported upper_level type: {upper_level}")
            assert observer_id in observers_names
            observer_index = observers_names.index(observer_id)
            assert 0 <= observer_index < len(observers_names)
            assert unit_id in units_names
            unit_index = units_names.index(unit_id)
            assert 0 <= unit_index <= len(units_names)
            for answer in _prepare_value(raw_answer):
                assert answer in possible_values, (answer, possible_values)
                answer_index = possible_values.index(answer)
                assert 0 <= answer_index < len(possible_values)
                answers_tensor[observer_index, unit_index, answer_index] += 1

    return _PreparedData(answers_tensor=answers_tensor,
                         possible_values=possible_values,
                         units_names=units_names,
                         observers_names=observers_names)


def from_list_of_lists(input_table: Iterable[Iterable[Union[RT, VT]]],
                       header: bool = Union[bool, Iterable[Union[OT, UT]]],
                       row_legend: Union[bool, List[Union[OT, UT]]] = False,
                       upper_level="observer",
                       value_constructor: Union[None, Callable[[RT], VT]] = None,
                       possible_values: Union[None, List[VT]] = None):
    """
    :param input_table: List-of-lists table containing observers' answers for each unit or containing None if specified
                        observer didn't evaluate for specified unit.
    :param upper_level: {"observer", "unit"}.
                        Specification of how input_table is composed.  If next(input_table.__iterator__) returns
                        information about different units for single observer the "observer" option have to be chosen.
                        If next(input_table.__iterator__) returns information about different observers answers for
                        single unit the "unit" option have to be chosen.
    :param header:
    :param row_legend:

    :param value_constructor: Optional function to be called on elements of input_table returning Hashable value to be
                               used. If None (default) elements of input_table will be used as is.
    :param possible_values: Optional list of possible values to be used for creating metric tensor in the future. If
                            None (default) the possible values will be derived from data.

    :return:
    """
    if upper_level not in {"observer", "unit"}:
        raise ValueError(f"Unsupported upper_level value {upper_level}")

    possible_values, track_value = init_possible_values(possible_values)

    index_table = []

    rows_iterator = iter(input_table)
    if isinstance(header, bool):
        if header:
            header = tuple(next(rows_iterator))
            if isinstance(row_legend, bool) and row_legend:
                header = header[1:]
            header_row_name = "header row"
        else:
            header = None
            header_row_name = "row 0"
    elif isinstance(header, Iterable):
        header = tuple(header)
        header_row_name = "passed header"
    else:
        raise TypeError(f"Unsupported header type: {header}")

    if isinstance(row_legend, bool):
        if row_legend:
            row_legend = []
            track_row_legend = True
        else:
            row_legend = None
            track_row_legend = False
    elif isinstance(row_legend, Iterable):
        row_legend = tuple(row_legend)
        track_row_legend = False
    else:
        raise TypeError(f"Unsupported column_legend type: {row_legend}")

    def _prepare_value(_raw_value) -> Union[None, VT]:
        if _raw_value is None:
            return None
        else:
            if value_constructor is None:
                return _raw_value
            else:
                return value_constructor(_raw_value)

    for row_index, row in enumerate(rows_iterator):
        items_iterator = iter(row)
        if track_row_legend:
            row_legend.append(next(items_iterator))
        index_row = []
        for column_index, raw_value in enumerate(items_iterator):
            try:
                value = _prepare_value(raw_value)
            except Exception as e:
                raise ValueError(f"Caught {type(e)} on constructor call during data preparation in "
                                 f"row: {row_index if row_legend is None else row_legend[row_index]}; "
                                 f"column: {column_index if header is None else header[column_index]}")
            if value is not None:
                if value in possible_values:
                    value_index = possible_values.index(value)
                else:
                    value_index = -1
            else:
                value_index = None
            if track_value:
                if value_index == -1:
                    index_row.append(len(possible_values))
                    possible_values.append(value)
                else:
                    index_row.append(value_index)

            else:
                if value_index == -1:
                    raise ValueError(f"Unexpected table value: {value}. Passed possible_values: {possible_values}")
                index_row.append(value_index)

        if header is not None and len(header) != len(index_row):
            raise ValueError(f"Rows have different number of elements.\n"
                             f"Row {row_index if row_legend is None else row_legend[row_index]} contains "
                             f"{len(index_row)} elements  but {header_row_name} contains {len(header)} elements")
        index_table.append(index_row)

    if not track_row_legend and row_legend is not None and len(row_legend) != len(index_table):
        raise ValueError(f"Passed row legend contains {len(row_legend)} "
                         f"but input_table contains {len(index_table)} rows")

    answers_tensor = numpy.array([[[1.0 if value_index == index else 0.0
                                    for index in range(len(possible_values))]
                                   for value_index in row]
                                  for row in index_table])

    if upper_level == "observer":
        observers_names = row_legend
        units_names = header
    elif upper_level == "unit":
        observers_names = header
        units_names = row_legend
        answers_tensor = answers_tensor.transpose((1, 0, 2))
    else:
        raise ValueError(f"Unsupported upper_level value {upper_level}")
    return _PreparedData(answers_tensor=answers_tensor,
                         possible_values=possible_values,
                         units_names=units_names,
                         observers_names=observers_names)
