"""
Utils for working with MLflow.

All classes & functions assume that both mlflow.set_tracking_uri() and
mflow.set_experiment() have been called beforehand to set the dir where all
expe are stored, and the expe itself. Note that some utilities will still
require to pass the experiment name/id or the tracking_uri.
"""
import mlflow
import tempfile
import os
import inspect
import json
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)


def create_expe(expe_name: str) -> str:
    """Create expe if does not exist, and return its id.

    Create the experiment if did not exist.

    Parameters
    ----------
    expe_name : str
        Name of the experiment

    Returns
    -------
    str
        Experiment ID (for mlflow functions)
    """
    experiment_id = get_expe_id(expe_name=expe_name)
    if experiment_id is None:
        logger.info(
            "Experiment did not exist. Created '{}'.".format(expe_name)
        )
        experiment_id = mlflow.create_experiment(name=expe_name)
    return experiment_id


def create_artifact_from_str(s: str, filename: str) -> None:
    """Create an artifact from a string

    The artifact name in MLflow will be `filename`.
    """
    with tempfile.TemporaryDirectory() as directory:
        artifact_tmp_path = os.path.join(directory, filename)
        with open(artifact_tmp_path, "w") as f:
            f.write(s)
        mlflow.log_artifact(artifact_tmp_path)


def params_to_query(params: dict) -> str:
    """Dictionary of parameters into MLFlow query

    Use with mlflow.search_runs(). Numeric parameters are turned into strings.

    Parameters
    ----------
    params : dict

    Returns
    -------
    str
    """
    query_terms = [
        "params.`" + k + "`" + '="' + str(v) + '"' for k, v in params.items()
    ]
    query = " and ".join(query_terms)
    return query


def run_w_params_exists(params: dict) -> bool:
    """Any previous run with given parameters?

    Checks whether there are runs with parameters `params` and status
    "FINISHED" in experiment `experiment_id`

    Parameters
    ----------
    params : dict
        params

    Returns
    -------
    bool
    """
    filter_str = params_to_query(params=params)
    filter_str += 'attributes.status = "FINISHED"'
    runs = mlflow.search_runs(
        filter_string=filter_str,
        run_view_type=mlflow.entities.ViewType.ALL,
    )
    exist = False if (runs.shape[0] == 0) else True
    return exist


def get_expe_id(expe_name: str) -> Union[str, None]:
    """Get MLflow experiment id based on its name.

    Parameters
    ----------
    expe_name : str
        expe_name

    Returns
    -------
    Union[str, None]
        str of id experiment if exists, otherwise None.
    """
    experiment = mlflow.get_experiment_by_name(expe_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = None
    return experiment_id


def dict_to_mlflow_params(dic: dict, concat_sep: str = "__") -> dict:
    """Convert configuration dict to mlflow parameters

    Perform two actions:
    - Collapse `dic` to one dimension, appending keys with `concat_sep`
    - Replace values of function type by their source string.

    Parameters
    ----------
    dic : dict
        dic
    concat_sep : str
        concat_sep

    Returns
    -------
    dict
    """
    out_dic = pd.io.json._normalize.nested_to_record(ds=dic, sep=concat_sep)
    for k, v in out_dic.items():
        if callable(v):
            out_dic[k] = inspect.getsource(v)
    return out_dic


def get_run_ids(experiment_id: str, max_results: int) -> list:
    """Get id of all runs associated to an experiment"""
    runs_info = mlflow.list_run_infos(
        experiment_id=experiment_id,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=max_results,
    )
    runs_id = [run_info.run_id for run_info in runs_info]
    return runs_id


def check_metrics_exist(tracking_uri: str, run_id: str, key: str) -> bool:
    """Check metric has been filled in the run

    Parameters
    ----------
    tracking_uri: str
        Folder in which experiments are stored
    run_id : str

    key : str
        Metric name

    Returns
    -------
    bool
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_metric_history(run_id=run_id, key=key)
        return True
    except Exception as e:
        if isinstance(e, mlflow.exceptions.MlflowException):
            return False
        else:
            raise TypeError("mlflow has risen an unexpected exception.")


def get_json_artifact(run_id: str, artifact_name: str) -> Union[list, dict]:
    """Get json artifact"""
    run = mlflow.get_run(run_id=run_id)
    json_path = os.path.realpath(
        os.path.join(run.info.artifact_uri, artifact_name)
    )
    with open(json_path) as f:
        artifact = json.load(f)
    return artifact


def get_run_params(run_id: str) -> dict:
    """Get parameters associated to an mlflow run"""
    run = mlflow.get_run(run_id=run_id)
    params = run.data.params
    return params
