# coding=utf-8
# Creation date: 14 нояб. 2020
# Creation time: 15:04
# Creator: SteamPeKa
import fractions
from typing import List


def list_of_lists_to_md_table(list_of_lists: List[List[str]]):
    column_index_to_max_len = []
    data_copy = []
    for row_index, row in enumerate(list_of_lists):
        row_copy = []
        data_copy.append(row_copy)
        for column_index, value in enumerate(row):
            assert isinstance(value, str), (value, type(value))
            value_copy = str(value)
            row_copy.append(value_copy)
            assert column_index <= len(column_index_to_max_len)
            if column_index == len(column_index_to_max_len):
                column_index_to_max_len.append(len(value_copy))
            else:
                column_index_to_max_len[column_index] = max(column_index_to_max_len[column_index], len(value_copy))
    column_index_to_format_string = [
        "{{:<{}}}".format(max_len) if column_index == 0 else "{{:>{}}}".format(max_len)
        for column_index, max_len in enumerate(column_index_to_max_len)
    ]
    placing_string = "|{}|".format("|".join(
        ":" + ("-" * (max_len - 1)) if column_index == 0 else ("-" * (max_len - 1)) + ":"
        for column_index, max_len in enumerate(column_index_to_max_len)
    ))
    for row in data_copy:
        if len(row) < len(column_index_to_max_len):
            row.extend([""] * (len(column_index_to_max_len) - len(row)))
    lines = [
        "|{}|".format("|".join(
            cell_format.format(cell_data) for cell_data, cell_format in zip(row, column_index_to_format_string)))
        for row in data_copy
    ]
    lines.insert(1, placing_string)
    return "\n".join(lines)


class HTMLNode(object):
    def __init__(self, tag, text, style):
        self.__tag = tag
        self.__text = text
        self.__style = style
        self.__children = []

    def add_child(self, child):
        assert isinstance(child, HTMLNode)
        self.__children.append(child)

    @property
    def tag(self):
        return self.__tag

    @property
    def text(self):
        return self.__text

    @property
    def style(self):
        return self.__style

    @property
    def children(self):
        return self.__children

    def __hash__(self):
        return hash((self.__tag,
                     self.__text,
                     self.__style))


def prepare_html_node(input_node: HTMLNode):
    node_to_representation = {input_node: None}
    while None in node_to_representation.values():
        current_keys = list(node_to_representation.keys())
        for node in current_keys:
            if len(node.children) == 0:
                node_to_representation[node] = "<{tag} style=\"{style}\">{text}</{tag}>".format(tag=node.tag,
                                                                                                text=node.text,
                                                                                                style=node.style)
            else:
                if all(node_to_representation.setdefault(child) is not None for child in node.children):
                    content = node.text + "\n" + "\n".join(node_to_representation[c] for c in node.children)
                    content = content.strip()
                    content = "\n    ".join(content.split("\n"))
                    node_to_representation[node] = "<{tag} style=\"{style}\">\n" \
                                                   "    {content}\n" \
                                                   "</{tag}>".format(tag=node.tag,
                                                                     style=node.style,
                                                                     content=content)
    return node_to_representation[input_node]


def list_of_lists_to_html_table(list_of_lists: List[List[str]], header=True, footer=True, left_legend=True,
                                right_legend=True, **kwargs):
    data_copy = []
    for row_index, row in enumerate(list_of_lists):
        row_copy = []
        data_copy.append(row_copy)
        for column_index, value in enumerate(row):
            assert isinstance(value, str), (value, type(value))
            value_copy = str(value)
            row_copy.append(value_copy)
    columns_count = max(len(r) for r in data_copy)
    for row in data_copy:
        if len(row) < columns_count:
            row.extend([""] * (columns_count - len(row)))
    common_style = ["border:1px solid black",
                    "border-collapse: collapse",
                    "font-size:small"]

    table_style = common_style + []
    table_style = ";".join(table_style)
    row_style = common_style + []
    row_style = ";".join(row_style)
    t_head_style = common_style + []
    t_head_style = ";".join(t_head_style)
    t_foot_style = common_style + []
    t_foot_style = ";".join(t_foot_style)
    t_h_style = common_style + []
    t_h_style = ";".join(t_h_style)
    td_style = common_style + []
    td_style = ";".join(td_style)
    tbody_style = common_style + []
    tbody_style = ";".join(tbody_style)

    table = HTMLNode("table", "", table_style)
    if header:
        head_row = HTMLNode("tr", "", row_style)
        head = HTMLNode("thead", "", t_head_style)
        head.add_child(head_row)
        table.add_child(head)
        for cell in data_copy.pop(0):
            head_row.add_child(HTMLNode("th", cell, t_h_style))
    if footer:
        foot_row = HTMLNode("tr", "", row_style)
        foot = HTMLNode("tfoot", "", t_foot_style)
        foot.add_child(foot_row)
        table.add_child(foot)
        for cell in data_copy.pop():
            foot_row.add_child(HTMLNode("th", cell, t_h_style))
    table_body = HTMLNode("tbody", "", tbody_style)
    table.add_child(table_body)
    for data_row in data_copy:
        table_row = HTMLNode("tr", "", row_style)
        table_body.add_child(table_row)
        for i, cell in enumerate(data_row):
            if (i == 0 and left_legend) or i == len(data_row) - 1 and right_legend:
                table_row.add_child(HTMLNode("th", cell, t_h_style))
            else:
                table_row.add_child(HTMLNode("td", cell, td_style))

    return prepare_html_node(table)


def fraction_to_str(fraction: fractions.Fraction, style="tex_math"):
    normalized_fraction = fractions.Fraction(fraction, 1, _normalize=True)
    numerator = normalized_fraction.numerator
    denominator = normalized_fraction.denominator
    if style == "tex_math":
        if denominator == 1:
            return "{:d}".format(numerator)
        else:
            return fr"\frac{{{numerator}}}{{{denominator}}}"
    if style == "tex_small":
        if denominator == 1:
            return "{:d}".format(numerator)
        else:
            return fr"{{}}^{{{numerator}}}{{/}}_{{{denominator}}}"
    else:
        raise NotImplementedError(f"style {style} is not implemented")


def prepare_markdown_header(header, level=4):
    return ("#" * level) + " " + str(header)
