import sklearn.model_selection
from typing import Dict


def generate_grid_from_conf(
    conf: Dict[str, list]
) -> sklearn.model_selection.ParameterGrid:
    """
    Return an interable parameter grid based on a dictionnary containing list
    of possible values for each key.
    """
    grid = sklearn.model_selection.ParameterGrid(param_grid=conf)
    return grid
