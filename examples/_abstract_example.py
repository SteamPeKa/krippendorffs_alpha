# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 15:58
# Creator: SteamPeKa

from ._pretty_prints import *


class AbstractExample(object):
    def __init__(self, general_format_data, main_observers_sequence, main_units_sequence,
                 main_possible_answers_sequence, no_value_placeholder=""):
        self.__main_observers_sequence = tuple(main_observers_sequence)
        assert all(isinstance(v, str) for v in self.__main_observers_sequence)
        assert len(self.__main_observers_sequence) == len(set(self.__main_observers_sequence))

        self.__main_units_sequence = tuple(main_units_sequence)
        assert all(isinstance(v, str) for v in self.__main_units_sequence)
        assert len(self.__main_units_sequence) == len(set(self.__main_units_sequence))

        self.__main_possible_answers_sequence = tuple(main_possible_answers_sequence)
        assert all(isinstance(v, str) for v in self.__main_possible_answers_sequence)
        assert len(self.__main_possible_answers_sequence) == len(set(self.__main_possible_answers_sequence))

        assert no_value_placeholder not in self.__main_possible_answers_sequence
        self.__no_value_placeholder = no_value_placeholder

        self.__general_format_data = {}
        assert isinstance(general_format_data, dict)
        for observer_id, unit_id_to_answer in general_format_data.items():
            assert isinstance(observer_id, str)
            assert observer_id in self.__main_observers_sequence
            assert isinstance(unit_id_to_answer, dict)
            for unit_id, raw_answer in unit_id_to_answer.items():
                assert isinstance(unit_id, str)
                assert unit_id in self.__main_units_sequence
                if isinstance(raw_answer, list):
                    for value in raw_answer:
                        assert isinstance(value, str)
                        assert value in self.__main_possible_answers_sequence
                        self.__general_format_data.setdefault(observer_id, {}).setdefault(unit_id, []).append(value)
                elif isinstance(raw_answer, str):
                    assert raw_answer in self.__main_possible_answers_sequence
                    self.__general_format_data.setdefault(observer_id, {}).setdefault(unit_id, []).append(raw_answer)
                else:
                    assert False
        assert len(set(self.__general_format_data.keys()) ^ set(self.__main_observers_sequence)) == 0
        seen_units_ids = set()
        for unit_id_to_answer in self.__general_format_data.values():
            for unit_id, answers_list in unit_id_to_answer.items():
                unit_id_to_answer[unit_id] = tuple(answers_list)
            seen_units_ids.update(unit_id_to_answer.keys())
        assert len(seen_units_ids ^ set(self.__main_units_sequence)) == 0

        self.____answers_table = None
        self.__markdown_answers_table = None
        self.__values_by_units_table = None
        self.__expected_coincidence_matrix = None
        self.____answers_frequencies = None
        self.__observer_semi_normed_coincidence_matrix = None
        self.__units_overlaps = None
        self.__evaluated_metric_tensors = {}
        self.__alpha_values = {}

    @property
    def observers_names(self):
        return tuple(self.__main_observers_sequence)

    @property
    def observers_count(self):
        return len(self.__main_observers_sequence)

    @property
    def units_names(self):
        return tuple(self.__main_units_sequence)

    @property
    def units_count(self):
        return len(self.__main_units_sequence)

    @property
    def possible_answers(self):
        return tuple(self.__main_possible_answers_sequence)

    @property
    def possible_answers_count(self):
        return len(self.__main_possible_answers_sequence)

    def __get_answers(self, observer_id, unit_id):
        return list(self.__general_format_data.get(observer_id, {}).get(unit_id, ()))

    @property
    def __answers_table(self):
        if self.____answers_table is None:
            result = ([[""] + list(self.units_names)] +
                      [([observer_id] + [",".join(self.__get_answers(observer_id, unit_id))
                                         for unit_id in self.units_names])
                       for observer_id in self.observers_names])
            self.____answers_table = result
        return self.____answers_table

    def get_answers_table(self, _format="raw"):
        if _format == "raw":
            return self.__answers_table
        elif _format == "html":
            return list_of_lists_to_html_table(self.__answers_table, left_legend=True,
                                               right_legend=False,
                                               header=True,
                                               footer=False)
        elif _format == "markdown":
            return list_of_lists_to_md_table(self.__answers_table)
        else:
            raise ValueError(f"Unknown format value: {_format}")

    def __metric_tensor(self, metric_name):
        if metric_name not in self.__evaluated_metric_tensors:
            if metric_name == "ratio":
                result = [[fractions.Fraction(0, 1) for __ in self.possible_answers] for _ in self.possible_answers]
                for c in range(self.possible_answers_count):
                    for k in range(c + 1, self.possible_answers_count):
                        if k > c:
                            result[c][k] = fractions.Fraction((c - k) ** 2, (c + k) ** 2)
            elif metric_name == "nominal":
                result = [[fractions.Fraction(0, 1) for __ in self.possible_answers] for _ in self.possible_answers]
                for c in range(self.possible_answers_count):
                    for k in range(self.possible_answers_count):
                        if k > c:
                            result[c][k] = fractions.Fraction(1, 1)
            elif metric_name == "interval":
                result = [[fractions.Fraction(0, 1) for __ in self.possible_answers] for _ in self.possible_answers]
                for c in range(self.possible_answers_count):
                    for k in range(self.possible_answers_count):
                        if k > c:
                            result[c][k] = fractions.Fraction((c - k) ** 2, 1)
            else:
                raise NotImplementedError(metric_name)
            self.__evaluated_metric_tensors[metric_name] = result
        return self.__evaluated_metric_tensors[metric_name]

    def summary(self, metrics=("nominal", "interval", "ratio"),
                headers_level=4, table_processor="html", formulas_style="tex_small"):
        def make_table(data, **kwargs):
            if table_processor == "html":
                return list_of_lists_to_html_table(data, **kwargs)
            elif table_processor == "markdown":
                return list_of_lists_to_md_table(data)
            else:
                raise NotImplementedError(f"Unknown table processor: {table_processor}")

        result_nodes = [
            "{header}\n\n{table}".format(header=prepare_markdown_header("Answers table", level=headers_level),
                                         table=self.get_answers_table(_format=table_processor))
        ]

        sums_per_units = [fractions.Fraction(0, 1) for _ in self.units_names]
        prepared_values_by_unit_table = [[""] + list(self.units_names) + [r"$n_{\bullet,v}$"]]
        overlaps = self.units_overlaps
        frequencies = []
        for possible_value, weights in zip(self.possible_answers, self.values_by_units_table):
            value_frequency = sum(w for w, o in zip(weights, overlaps) if o > 1)
            frequencies.append(value_frequency)
            prepared_values_by_unit_table.append(
                [possible_value] + list("${}$".format(fraction_to_str(weight, style=formulas_style))
                                        for weight in weights) + [
                    "${}$".format(fraction_to_str(value_frequency, style=formulas_style))]
            )
            for i in range(self.units_count):
                sums_per_units[i] += weights[i]
        prepared_values_by_unit_table.append(
            ["${}$".format(r"n_{u,\bullet}")] + list("${}$".format(fraction_to_str(weight, style=formulas_style))
                                                     for weight in sums_per_units) + [
                "${}$".format(fraction_to_str(sum(frequencies), style=formulas_style))]
        )
        result_nodes.append("{header}\n\n{table}".format(
            header=prepare_markdown_header("Value by unit weight table", level=headers_level),
            table=make_table(prepared_values_by_unit_table)
        ))

        expected_not_normed_coincidence_table = [[""] + list(self.possible_answers)]
        for row_index, row in enumerate(self.expected_not_normed_coincidence_matrix):
            expected_not_normed_coincidence_table.append([self.possible_answers[row_index]] + [
                "${}$".format(fraction_to_str(value, style=formulas_style))
                for value in row
            ])
        result_nodes.append("{header}\n"
                            "The upper-diagonal matrix with elements $g_{{c,k}}=n_{{\\bullet,c}}\cdot n_{{\\bullet,c}}$\n\n"
                            "\n\n{table}".format(
            header=prepare_markdown_header("Expected not normed coincidence matrix", level=headers_level),
            table=make_table(expected_not_normed_coincidence_table, left_legend=True, right_legend=False,
                             header=True, footer=False)
        ))

        observed_semi_normed_coincidence_table = [[""] + list(self.possible_answers)]
        for row_index, row in enumerate(self.observer_semi_normed_coincidence_matrix):
            observed_semi_normed_coincidence_table.append([self.possible_answers[row_index]] + [
                "${}$".format(fraction_to_str(value, style=formulas_style))
                for value in row
            ])

        result_nodes.append("{header}\n"
                            "The upper-diagonal matrix with elements "
                            "$h_{{c,k}}=\\sum\\limits_{{u}}{{\\frac{{n_{{u,c}}\cdot n_{{u,c}}}}"
                            "{{n_{{u,\\bullet}}-1}}}}$\n\n"
                            "{table}".format(
            header=prepare_markdown_header("Observed semi-normed coincidence matrix", level=headers_level),
            table=make_table(observed_semi_normed_coincidence_table, left_legend=True, right_legend=False,
                             header=True, footer=False)
        ))
        if len(metrics) != 0:
            result_nodes.append(prepare_markdown_header("Values for some metrics", level=headers_level))
        for metric_name in metrics:
            metric_tensor = [[""] + list(self.possible_answers)]
            for row_index, row in enumerate(self.__metric_tensor(metric_name)):
                metric_tensor.append([self.possible_answers[row_index]] + [
                    "${}$".format(fraction_to_str(value, style=formulas_style))
                    for value in row
                ])
            fraction_result, decimal_result, result_for_tex = self.__evaluate_alpha(metric_name,
                                                                                    fraction_style=formulas_style)

            result_nodes.append(("{header}\n"
                                 "The upper-diagonal distance matrix for this metric is presented below\n\n"
                                 "{table}\n\n"
                                 "Using it and two coincidence matrices above it is easy co calculate alpha value:\n"
                                 "$$\n"
                                 "{alpha_value}\n"
                                 "$$\n\n").format(
                header=prepare_markdown_header("Value for {} metric".format(metric_name), level=headers_level + 1),
                table=make_table(metric_tensor, left_legend=True, right_legend=False,
                                 header=True, footer=False),
                alpha_value=result_for_tex
            ))

        return "\n\n".join(result_nodes)

    @property
    def values_by_units_table(self):
        if self.__values_by_units_table is None:
            result = []
            for possible_value in self.possible_answers:
                row = []
                result.append(row)
                for unit_id in self.units_names:
                    sum_weight = fractions.Fraction(0, 1)
                    for observer_id in self.observers_names:
                        answers = self.__get_answers(observer_id, unit_id)
                        if len(answers) > 0:
                            weight_denominator = len(answers)
                            weight_numerator = sum(1 if v == possible_value else 0 for v in answers)
                            sum_weight += fractions.Fraction(weight_numerator, weight_denominator)
                    row.append(sum_weight)
            self.__values_by_units_table = result
        return self.__values_by_units_table

    @property
    def __answers_frequencies(self):
        if self.____answers_frequencies is None:
            result = [sum(v for v, o in zip(row, self.units_overlaps) if o > 1) for row in self.values_by_units_table]
            self.____answers_frequencies = result
        return self.____answers_frequencies

    @property
    def expected_not_normed_coincidence_matrix(self):
        if self.__expected_coincidence_matrix is None:
            result = [[fractions.Fraction(0, 1) for __ in self.possible_answers] for _ in self.possible_answers]
            for c in range(self.possible_answers_count):
                for k in range(self.possible_answers_count):
                    if k > c:
                        result[c][k] = self.__answers_frequencies[c] * self.__answers_frequencies[k]
            self.__expected_coincidence_matrix = result
        return self.__expected_coincidence_matrix

    @property
    def observer_semi_normed_coincidence_matrix(self):
        if self.__observer_semi_normed_coincidence_matrix is None:
            result = [[fractions.Fraction(0, 1) for __ in self.possible_answers] for _ in self.possible_answers]
            for c in range(self.possible_answers_count):
                for k in range(self.possible_answers_count):
                    if k > c:
                        for u in range(self.units_count):
                            if self.units_overlaps[u] > 1:
                                result[c][k] += ((self.values_by_units_table[c][u] * self.values_by_units_table[k][u]) /
                                                 (self.units_overlaps[u] - fractions.Fraction(1, 1)))
            self.__observer_semi_normed_coincidence_matrix = result
        return self.__observer_semi_normed_coincidence_matrix

    @property
    def units_overlaps(self):
        if self.__units_overlaps is None:
            result = [fractions.Fraction(0, 1) for _ in range(self.units_count)]
            for value_index, values_per_unit in enumerate(self.values_by_units_table):
                for unit_index, value in enumerate(values_per_unit):
                    result[unit_index] += value
            self.__units_overlaps = result
        return self.__units_overlaps

    @property
    def total_weight(self):
        return sum(o for o in self.__units_overlaps if o > 1)

    def alpha_value(self, metric_name):
        fraction_result, decimal_result, result_for_tex = self.__evaluate_alpha(metric_name)
        return fraction_result

    def __evaluate_alpha(self, metric_name, fraction_style="small_tex"):
        if (metric_name, fraction_style) not in self.__alpha_values:
            D_o_tex = []
            D_e_tex = []

            D_e_fraction = []
            D_o_fraction = []
            metric_tensor = self.__metric_tensor(metric_name)

            for c in range(self.possible_answers_count):
                for k in range(c + 1, self.possible_answers_count):
                    distance_fraction = metric_tensor[c][k]
                    need_distance = distance_fraction.numerator != distance_fraction.denominator
                    distance_tex = fraction_to_str(distance_fraction, style=fraction_style)
                    expected_weight = self.expected_not_normed_coincidence_matrix[c][k]
                    observed_weight = self.observer_semi_normed_coincidence_matrix[c][k]

                    if expected_weight != 0:
                        D_e_fraction.append(distance_fraction * expected_weight)
                    if observed_weight != 0:
                        D_o_fraction.append(distance_fraction * observed_weight)

                    if need_distance:
                        if observed_weight != 0:
                            D_o_tex.append(
                                (fr"{distance_tex}"
                                 fr"\cdot"
                                 fr"{fraction_to_str(observed_weight, style=fraction_style)}")
                            )
                        if expected_weight != 0:
                            D_e_tex.append(
                                (fr"{distance_tex}"
                                 fr"\cdot"
                                 fr"{fraction_to_str(expected_weight, style=fraction_style)}")
                            )

                    else:
                        if observed_weight != 0:
                            D_o_tex.append(fraction_to_str(observed_weight, style=fraction_style))
                        if expected_weight != 0:
                            D_e_tex.append(fraction_to_str(expected_weight, style=fraction_style))

            D_o_tex = " + ".join(D_o_tex)
            D_e_tex = " + ".join(D_e_tex)

            D_o_fraction = sum(D_o_fraction)
            D_e_fraction = sum(D_e_fraction)

            n = self.total_weight
            fraction_result = fractions.Fraction(1, 1) - (
                    (fractions.Fraction(n, 1) - fractions.Fraction(1, 1)) * (D_o_fraction / D_e_fraction))
            assert isinstance(fraction_result, fractions.Fraction)
            decimal_result = "{:7.5f}".format(float(fraction_result))
            formula = r"1- (n_{\bullet,\bullet}-1)\frac{\sum\limits_{c}\sum\limits_{k}\delta^{2}_{c,k}\cdot h_{c,k}}" \
                      r"{\sum\limits_{c}\sum\limits_{k}\delta^{2}_{c,k}g_{c,k}}"
            result_for_tex = fr"""\begin{{align*}}\
\alpha_{{\text{{{metric_name}}}}} &= {formula} \\
 &= 1 - ({n}-1) \cdot\frac{{ {D_o_tex} }}{{ {D_e_tex} }} = \frac{{{fraction_result.numerator}}}{{{fraction_result.denominator}}} \approx  {decimal_result}
\end{{align*}}"""
            self.__alpha_values[(metric_name, fraction_style)] = fraction_result, decimal_result, result_for_tex
        return self.__alpha_values[(metric_name, fraction_style)]
