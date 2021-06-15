"""
Tools for grading
"""
import copy
import os
import logging
import mlflow
from dataclasses import dataclass
from typing import Dict

import generatools.sequences
import generatools.utils.text

logger = logging.getLogger(__name__)


@dataclass
class LikertScale:
    """
    Likert scale

    Parameters
    ----------
    name: str
        Name of the metric
    level: str
        At what level should we use this scale? One of ["sequence", "prompt"]. Value
        "prompt" means we give an evaluation of all the sequences associated to a
        new datapoint.
    description: str
        Description of the likert scale, preceding the items.
    items: Dict[int, str]
        Pair of value, description of value
    """

    name: str
    level: str
    description: str
    items: Dict[int, str]

    def get_description(self):
        concat_items = ", ".join(
            [str(k) + ": '" + v + "'" for k, v in self.items.items()]
        )
        return self.description + "\n" + concat_items

    def item_with_check(self, item: int):
        """
        Failsafe. Return the item if belongs to items.keys(), raise a ValueError
        else.
        """
        items_keys = self.items.keys()
        if item in items_keys:
            return item
        else:
            raise ValueError(
                "Item key '{}' not on the scale ({})".format(item, items_keys)
            )

    def __post_init__(self):
        """Check 'level' is among expected values"""
        allowed_levels = ["sequence", "prompt"]
        if self.level not in allowed_levels:
            raise ValueError(
                "'level' should be one of {}.".format(allowed_levels)
            )


class PrintRunAnalysisHeader(object):
    """
    Print an intro to the grading of generated sequences associated to a run.

    Print a header (with run id if any) and the parameters used for generation.
    User must call the instance to print the header.
    """

    def __call__(self, run_name: str = None, params: str = None):
        """Generate the introduc

        Parameters
        ----------
        run_name : str
            mlflow run name
        params : str
            Generation parameters
        """
        self._clear_screen()
        if run_name is not None:
            grading_intro = self._header_w_run_id(run_id=run_name)
        else:
            grading_intro = self._header_wo_run_id(run_id=run_name)
        if params is not None:
            grading_intro += "\n"
            grading_intro += self._format_params(params=params)
            grading_intro += "\n"
        print(grading_intro)

    def _clear_screen(self):
        if os.name == "posix":  #  Linux + OSX
            _ = os.system("clear")
        else:  # Windows
            _ = os.system("cls")

    def _header_wo_run_id(self, run_id: str = None):
        run_header_str = "RUN ANALYSIS"
        return run_header_str

    def _header_w_run_id(self, run_id: str = None):
        run_header_str = "RUN " + run_id
        run_header_str = (
            "=" * len(run_header_str)
            + "\n"
            + run_header_str
            + "\n"
            + "=" * len(run_header_str)
        )
        return run_header_str

    def _format_params(self, params: dict = None):
        params_str = "Parameters: " + str(params)
        return params_str


class SeqsHandGrading(object):
    """
    Grade sequences associated to a given prompt with a specified metric.

    Use method `interactive_grading`.
    """

    def __init__(self):
        pass

    def interactive_grading(
        self,
        prompt_seqs_pair: generatools.sequences.PromptSeqsPair,
        metric: LikertScale,
        show_raw_seq: bool = False,
    ) -> generatools.sequences.PromptSeqsPair:
        """Interactive grading sequences associated to a given prompt with a specified metric.

        The modalities for the grading are defined in `metric`:
        - if metric.level is "prompt", one grade will be set for all sequences
          associated to a prompt. If it is "sequence", there will be one grade
          for every sequence.
        - Accepted grades are the one set in  `metric.items.keys()`.

        Parameters
        ----------
        prompt_seqs_pair : generatools.sequences.PromptSeqsPair
            (prompt, generated sequences) pair to grade
        metric : LikertScale
            metric used for grading
        show_raw_seq : bool
            Display raw sequences in the grading screen?

        Returns
        -------
        generatools.sequences.PromptSeqsPair
            Equal to `prompt_seqs_pair`, with filling for the prompt_lvl_eval
            (resp seq_lvl_eval) attribute when metric.level is "prompt"
            ("sequence")
        """
        prompt_seqs_pair = copy.deepcopy(prompt_seqs_pair)
        # Grading
        if metric.level == "prompt":
            prompt_seqs_pair_w_grades = self._grading_at_prompt_level(
                prompt_seqs_pair=prompt_seqs_pair,
                metric=metric,
                show_raw_seq=show_raw_seq,
            )
        elif metric.level == "sequence":
            prompt_seqs_pair_w_grades = self._grading_at_sequence_level(
                prompt_seqs_pair=prompt_seqs_pair,
                metric=metric,
                show_raw_seq=show_raw_seq,
            )
        return prompt_seqs_pair_w_grades

    def _grading_at_prompt_level(
        self,
        prompt_seqs_pair: generatools.sequences.PromptSeqsPair,
        metric: LikertScale,
        show_raw_seq: bool = False,
    ) -> generatools.sequences.PromptSeqsPair:
        """Grade at once all sequences associated to one prompt"""
        n_seqs = len(prompt_seqs_pair.sequences)
        header = self._format_keyword_header(prompt_seqs_pair=prompt_seqs_pair)
        print(header)
        for seq_idx in range(n_seqs):
            # Prepare str
            trimmed_seq_str = "- Trimmed sequence {}/{}: '{}'".format(
                seq_idx + 1,
                n_seqs,
                prompt_seqs_pair.sequences_trimmed[seq_idx],
            )
            raw_seq_str = "- Raw sequence {}/{}: '{}'".format(
                seq_idx + 1, n_seqs, prompt_seqs_pair.sequences[seq_idx]
            )
            # Print and query
            print(trimmed_seq_str)
            if show_raw_seq:
                print(raw_seq_str)
        scale_desc_str = metric.get_description()
        print("\n" + scale_desc_str)
        # Get grade
        grade = self._grade_inputer(metric=metric)
        prompt_seqs_pair.prompt_lvl_eval[metric.name] = grade
        prompt_seqs_pair.check_attributes_sanity()
        return prompt_seqs_pair

    def _grading_at_sequence_level(
        self,
        prompt_seqs_pair: generatools.sequences.PromptSeqsPair,
        metric: LikertScale,
        show_raw_seq: bool = False,
    ):
        """Grade each sequence separately"""
        raise NotImplementedError(
            "Grading at sequence level not implemented yet."
        )

    def _grade_inputer(self, metric: LikertScale):
        """
        Inputing row for in the grading screen. Make sure the grade is valid.
        """
        successful_grading = False
        while not successful_grading:
            try:
                inp = generatools.utils.text.str_to_digit(
                    input("\n>> Chose one item: ")
                )
                grade = metric.item_with_check(inp)
                successful_grading = True
            except Exception as e:
                print(e)
        return grade

    def _format_keyword_header(
        self, prompt_seqs_pair: generatools.sequences.PromptSeqsPair
    ):
        """
        Header for grading at the prompt level. Shows the keywords (an empty
        set if no keyword).
        """
        keyword_header_str = (
            "\n==== KEYWORD SET {}: ".format(prompt_seqs_pair.keywords)
            + "===="
        )
        return keyword_header_str


def interactive_grading_from_mlflow_experiment(
    conf: dict,
    metric: LikertScale,
) -> None:
    """
    Grade generated sequences in an mlflow experiment

    Grading is performed only on PromptSeqsPair for which ther metric
    scale.name is missing.

    Grades will be stored in mlflow in the PromptSeqsPair objects.

    Parameters
    ----------
    conf : dict
        Configuration used with
        generatools.genseqs.run_grid_generation_from_conf when generating and
        storing sequences in mlflow.
    metric : generatools.grading.LikertScale
        scale to be used as a metric.

    Returns
    -------
    None
    """
    # Set experiment for all mlflow commands
    mlflow.set_tracking_uri(uri=conf["mlflow_expe_dirpath"])
    # Set experiment for all mlflow commandso
    mlflow.set_experiment(
        experiment_name=conf["mlflow_expe_name"],
    )
    #  Get experiment id
    experiment_id = generatools.utils.mlflow.create_expe(
        expe_name=conf["mlflow_expe_name"],
    )

    # Get ids of all runs
    run_ids = generatools.utils.mlflow.get_run_ids(
        experiment_id=experiment_id, max_results=1000
    )

    # Get client
    client = mlflow.tracking.MlflowClient(
        tracking_uri=conf["mlflow_expe_dirpath"]
    )

    # For each run
    for run_id in run_ids:
        # If metric already set, go to next run
        metric_exists = generatools.utils.mlflow.check_metrics_exist(
            tracking_uri=conf["mlflow_expe_dirpath"],
            run_id=run_id,
            key=metric.name,
        )
        if metric_exists:
            continue
        # Get sequences
        prompt_seqs_pair_list = generatools.utils.mlflow.get_json_artifact(
            run_id=run_id, artifact_name=conf["mlflow_results_json_name"]
        )
        prompt_seqs_pair_list = [
            generatools.sequences.PromptSeqsPair(**prompt_seqs_pair)
            for prompt_seqs_pair in prompt_seqs_pair_list
        ]
        # Get parameters
        params = generatools.utils.mlflow.get_run_params(run_id=run_id)
        # Print header for this run
        generatools.grading.PrintRunAnalysisHeader()(
            run_name=run_id, params=params
        )
        # Associate grade to sequences
        for seq_idx in range(len(prompt_seqs_pair_list)):
            # Add grade to sequences
            prompt_seqs_pair = prompt_seqs_pair_list[seq_idx]
            grader = generatools.grading.SeqsHandGrading().interactive_grading
            prompt_seqs_pair = grader(
                prompt_seqs_pair=prompt_seqs_pair,
                metric=metric,
                show_raw_seq=False,
            )
            prompt_seqs_pair_list[seq_idx] = prompt_seqs_pair
        prompt_seqs_pair_list = generatools.sequences.PromptSeqsPairsList(
            prompt_seqs_pair_list
        )
        # Calculate overall metrics
        if metric.level == "prompt":
            metrics = prompt_seqs_pair_list.average_prompt_lvl_metrics()
        elif metric.level == "sequence":
            metrics = prompt_seqs_pair_list.average_seq_lvl_metrics()
        # Log metrics
        for metric_name, metric_val in metrics.items():
            client.log_metric(
                run_id=run_id,
                key=metric_name,
                value=metric_val,
            )
        # Save grade with sequences
        # 1/ Remove previous dict
        run = client.get_run(run_id)
        dict_path = os.path.join(
            run.info.artifact_uri, conf["mlflow_results_json_name"]
        )
        try:  # Works only if there is something to remove
            os.remove(dict_path)
            logger.debug("Removed previously existing {}".format(dict_path))
        except Exception:
            pass
        # 2/ Use new dict
        prompt_seqs_pair_list_json = prompt_seqs_pair_list.to_json()
        client.log_dict(
            run_id=run_id,
            dictionary=prompt_seqs_pair_list_json,
            artifact_file=conf["mlflow_results_json_name"],
        )
