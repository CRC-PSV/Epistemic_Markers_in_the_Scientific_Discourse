"""Util functions to manage command line inputs"""
from typing import Sequence
import json, csv


def read_validate_input(prompt: str, values: Sequence[str], success_msg: str = None, error_msg: str = None, to_lower: bool = True):
    """Prompts for an input based on a list of accepted values. Loops until a valid value is entered.

    Args:
        prompt: Prompt message to display, directly passed to input().
        values: A list (or list-like object) of accepted input values.
        success_msg: Message to print after a valid input is entered. Leaving it as None will not print anything, passing an empty string will print a blank line.
        error_msg: Message to print after an invalid input is entered. Leaving it as None will not print anything, passing an empty string will print a blank line.
        to_lower: Whether to call .lower() on the input before comparing it to values.

    Returns:
        The first valid value entered. Will be transformed to lowercase if to_lower is set to True.
    """

    while True:
        s = input(prompt).lower() if to_lower else input(prompt)
        if s in values:
            if success_msg is not None:
                print(success_msg)
            return s
        else:
            if error_msg is not None:
                print(error_msg)


def read_y_n_input(prompt: str = 'Continue? (y/n): ') -> bool:
    """Asks for a y/n input and wait for a valid input

    Accepts 'y', 'n', 'yes', 'no', case insensitive.

    Args:
        prompt: The prompt message to display

    Returns:
        The boolean corresponding to the input value
    """

    s = read_validate_input(prompt, ['y', 'n', 'yes', 'no'])

    return s[0] == 'y'


def load_json(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def make_list_mapping_from_csv_path(csv_path):
    """Reads a csv and makes a key: [values] mapping

    Adds one entry per row, using the first column as the key and the rest as a value list. Ignores rows with no value
    in the first column.
    """

    f = open(csv_path, newline='')
    d = {n[0]: [n[i + 1] for i in range(len(n) - 1) if n[i + 1] != ''] for n in csv.reader(f)}

    # Remove categories with no associated value
    d = {cat: words for cat, words in d.items() if len(words) > 0}

    return d


def load_csv_values_as_single_list(csv_path, sort_values=True):

    mapping = make_list_mapping_from_csv_path(csv_path)
    values = list({w for li in mapping.values() for w in li})

    return sorted(values) if sort_values else values


if __name__ == '__main__':
    pass
    # print(Path.cwd())
