import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from numbers import Number


@dataclass
class PromptSeqsPair:
    """
    Store pairs of (sequence seed, list of sequence endings), together with
    their evaluation.

    Data storage when for a given sequence seed (prompt) that may or may not be
    characterized by a set of keywords (keywords), we generate one or
    more sequence endings (sequences) and optionally their cleaned counterpart
    (sequences_trimmed).

    We can associate an overall set of eval (prompt_lvl_eval) of the form
    {metric_name1: metric1, metric_name2: metric2, ...}.

    We can also associate to each sequence a set of eval (seq_lvl_eval).
    seq_lvl_eval is then a dict of eval_name: [metric_seq1, metric_seq2, ...].

    Use check_attributes_sanity() to make sure all attributes are consistent in
    length and types (done automatically at init, but not afterward.)
    """

    prompt: str
    sequences: List[str]
    keywords: Optional[list] = None
    sequences_trimmed: Union[List[str], None] = None
    prompt_lvl_eval: Dict[str, Number] = field(default_factory=dict)
    seq_lvl_eval: Dict[str, List[Number]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Transform object to dict"""
        return self.__dict__

    def average_seq_lvl_eval(self) -> dict:
        """For each metric, return its average value across sequences"""
        if self.seq_lvl_eval is None:
            raise ValueError("Cannot average 'seq_lvl_eval' when None.")
        mean_dict = {}
        for metric_name, metric_values in self.seq_lvl_eval.items():
            metric_mean = statistics.mean(self.seq_lvl_eval[metric_name])
            mean_dict[metric_name] = metric_mean
        return mean_dict

    def __post_init__(self):
        self.check_attributes_sanity()

    def check_attributes_sanity(self):
        """Check arguments consistency.

        Checked:
        - sequences and sequences_trimmed have same lenght (if specified)
        - Check prompt_lvl_eval and seq_lvl_eval types and lengths (if specified)
        """
        if self.sequences_trimmed is not None:
            if len(self.sequences_trimmed) != len(self.sequences):
                raise ValueError(
                    "Lengths of sequences and sequences_trimmed differ."
                )
        # Check prompt_lvl_eval:
        if len(self.prompt_lvl_eval) != 0:
            if not isinstance(self.prompt_lvl_eval, dict):
                raise TypeError("prompt_lvl_eval should be a dict")
            for metric_name, metric in self.prompt_lvl_eval.items():
                if not isinstance(metric, Number):
                    raise TypeError(
                        "values in prompt_lvl_eval should be a numbers.Number"
                    )
        # Check seq_lvl_eval
        if len(self.seq_lvl_eval) != 0:
            if not isinstance(self.seq_lvl_eval, dict):
                raise TypeError("prompt_lvl_eval should be a dict")
            for metric_name, metric in self.seq_lvl_eval.items():
                if not isinstance(metric, list):
                    raise TypeError("values in seq_lvl_eval should be a list")
                if len(metric) != len(self.sequences):
                    raise ValueError(
                        "values in seq_lvl_eval should be have same length as sequences"
                    )


class PromptSeqsPairsList(object):
    """
    List of PromptSeqsPair objects.

    Two important methods:
    - average_prompt_lvl_metrics: to average metrics at the prompt level
    - average_seq_lvl_metrics: to average metrics at the prompt level
    """

    def __init__(self, ls: List[PromptSeqsPair]):
        self._list = ls
        self._check_all_are_prompt_seqs_pair(ls=self._list)

    def _check_all_are_prompt_seqs_pair(self, ls):
        for prompt_seqs_pair in ls:
            if not isinstance(prompt_seqs_pair, PromptSeqsPair):
                raise ValueError("All elements in ls should be PromptSeqsPair")

    def _check_shared_metrics_names(self, ls, lvl):
        """lvl is one of 'prompt_lvl_eval', 'seq_lvl_eval'"""
        metrics_names = getattr(self._list[0], lvl).keys()
        for i in range(1, len(self._list)):
            if getattr(self._list[i], lvl).keys() != metrics_names:
                raise KeyError(
                    "Not all PromptSeqsPair in this list have the same metrics listed"
                )

    def __getitem__(self, item) -> PromptSeqsPair:
        """Allows looping, slicing etc. on self"""
        return self._list[item]

    def to_json(self) -> List[dict]:
        """
        Transform the list of prompt_seqs_pair into a list of dict
        """
        out_json = [
            prompt_seqs_pair.to_dict() for prompt_seqs_pair in self._list
        ]
        return out_json

    def average_prompt_lvl_metrics(
        self,
    ) -> dict:
        """Average prompt level metrics across all pairs"""
        metric_averages = {}
        self._check_shared_metrics_names(ls=self._list, lvl="prompt_lvl_eval")
        metrics_names = self._list[0].prompt_lvl_eval.keys()
        for metric_name in metrics_names:
            metric_values = [
                self._list[i].prompt_lvl_eval[metric_name]
                for i in range(len(self._list))
            ]
            avg_metric = statistics.mean(metric_values)
            metric_averages[metric_name] = avg_metric
        return metric_averages

    def average_seq_lvl_metrics(self) -> dict:
        """Average prompt level metrics across all pairs"""
        metric_averages = {}
        self._check_shared_metrics_names(ls=self._list, lvl="seq_lvl_eval")
        metrics_list = [
            prompt_seqs_pair.average_seq_lvl_eval()
            for prompt_seqs_pair in self._list
        ]
        metrics_names = metrics_list[0].keys()
        for metric_name in metrics_names:
            metric_values = [
                metrics_list[i][metric_name] for i in range(len(self._list))
            ]
            metric_mean = statistics.mean(metric_values)
            metric_averages[metric_name] = metric_mean
        return metric_averages
