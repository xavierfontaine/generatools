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
from typing import Union, Dict

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


def run_w_params_exists(
    params: dict, experiment_id: str, params_artifact_name: str
) -> bool:
    """Any previous run with given parameters?

    Checks whether there are non-deleted runs with parameters `params` and
    status "FINISHED" in experiment `experiment_id`.

    Note this is done wrt to parameters stored by the user as json
    in `params_artifact_name`.

    Parameters
    ----------
    params : dict
        params

    experiment_id : dict
        experiment_id

    Returns
    -------
    bool
    """
    all_runs_params = get_json_artifact_for_all_runs(
        experiment_id=experiment_id, artifact_name=params_artifact_name
    )
    similar_run_found = any([params == p for p in all_runs_params.values()])
    return similar_run_found


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
    runs_id = [
        run_info.run_id
        for run_info in runs_info
        if run_info.status == "FINISHED"
    ]
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


def log_json_artifact(json_dict: dict, filename: str) -> None:
    """Avoid using mlflow.log_json which is considered experimental"""
    json_str = json.dumps(obj=json_dict)
    create_artifact_from_str(s=json_str, filename=filename)


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
    """
    Get parameters associated to an mlflow run

    NOTE: this will retrieve the parameters from the params slot of mlflow
    (which str-ify all parameters). Consider storing and loading parameters
    from a json instead.
    """
    run = mlflow.get_run(run_id=run_id)
    params = run.data.params
    return params


def get_json_artifact_for_all_runs(
    experiment_id: str, artifact_name: str
) -> Dict[str, dict]:
    """
    Get json artifact associated to each run in a experiment

    Retrieve only for runs that are not deleted and
    considered "FINISHED"

    Parameters
    ----------
    experiment_id : str
        experiment_id

    artifact_name: str
        Name of the artifact (as was saved beforehand)

    Returns
    -------
    Dict[str, dict]
        dict with keys run id, values parameters
    """
    runs_info = mlflow.list_run_infos(
        experiment_id=experiment_id,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000000,
    )
    all_artifacts = {}
    for run_info in runs_info:
        run_id = run_info.run_id
        # If status is FINISHED, then fetch the params
        if run_info.status == "FINISHED":
            artifact = get_json_artifact(
                run_id=run_id, artifact_name=artifact_name
            )
            all_artifacts[run_id] = artifact
    return all_artifacts
