import pathlib
from typing import Any, Union
import json


def load_json(path: Union[str, pathlib.Path]) -> list | dict:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_root_folder() -> pathlib.Path:
    return pathlib.Path.cwd()


def get_hydra_root_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_statement_variable(selected_code: str, local_variables: dict[str, Any]) -> list[str]:
    """
    INPUT:
        selected_code: Selected execution code
        local_variables: list of LOCAL VARIABLE (delete after each query finished)

    OUTPUT:
        selected code and probability
    """
    executed_variable_list = []

    for one_row_code in selected_code.split('\n'):
        if ' = ' in one_row_code: # has variable
            variable_name = one_row_code.split(' = ')[0] # get variable name --str type
            if variable_name in local_variables:
                executed_variable_list.append(variable_name) # append local variable into list.

    return executed_variable_list
