"""
Text preprocessing
"""
import copy
from typing import List, Optional


def prompt_formatter(
    intro: str = "",
    kws: Optional[List[List[str]]] = None,
    txts: Optional[List[str]] = None,
    ex_sep: str = "",
    ex_add_numeral: bool = False,
    ex_postnum_starter: str = "",
    ex_kws_lhs: str = "",
    ex_kws_sep: str = "",
    ex_kws_rhs: str = "",
    ex_txts_lhs: str = "",
    ex_txts_rhs: str = "",
    remove_right_trailing_space: bool = False,
) -> str:
    """Make prompt

    The prompt can contain three categories of elements: an intro, keywords,
    and texts. All of them can be left unspecified. See 'Examples' below for
    understanding usage and consequences.

    If both are specified, kws should be one element longer than txts (see
    example)

    Parameters
    ----------
    intro : str
        intro
    kws : Optional[List[List[str]]]
        kws
    txts : Optional[List[str]]
        txts
    ex_sep : str
        ex_sep
    ex_add_numeral : bool
        ex_add_numeral
    ex_postnum_starter : str
        ex_postnum_starter
    ex_kws_lhs : str
        ex_kws_lhs
    ex_kws_sep : str
        ex_kws_sep
    ex_kws_rhs : str
        ex_kws_rhs
    ex_txts_lhs : str
        ex_txts_lhs
    ex_txts_rhs : str
        ex_txts_rhs
    remove_right_trailing_space : bool
        remove_right_trailing_space

    Returns
    -------
    str
        Prompt

    Example
    -------
    >>> prompt_formatter(
    ...     intro="Generate sentences with keywords.\n",
    ...     txts=["The table is yellow", "The car is blue"],
    ...     kws=[
    ...         ["table", "yellow"],
    ...         ["car", "blue"],
    ...         ["house", "red"],
    ...     ],
    ...     ex_sep="\n",
    ...     ex_add_numeral=True,
    ...     ex_postnum_starter=". ",
    ...     ex_kws_sep=", ",
    ...     ex_kws_lhs="Keywords: \"",
    ...     ex_kws_rhs="\"",
    ...     ex_txts_lhs=" Sentence: \"",
    ...     ex_txts_rhs=".\"",
    ...     remove_right_trailing_space=True,
    ... )
    'Generate sentences with keywords.\n1. Keywords: "table, yellow" Sentence: "The table is yellow."\n2. Keywords: "car, blue" Sentence: "The car is blue."\n3. Keywords: "house, red" Sentence: "'
    """
    # Sanity on lengths of kws & ex (kws should be one longer than txts)
    if (kws is not None) and (txts is not None):
        if len(kws) != len(txts) + 1:
            raise ValueError(
                "kws should be one element longer than txts"
                f" (len of kws: {len(kws)}, of txt: {len(txts)})"
            )
    # kws
    examples = ""
    if kws is not None:
        # Sanity: all elements should be list
        if not all([isinstance(kw, list) for kw in kws]):
            raise ValueError("All elements in kw should be lists")
        kws_w_lhs_rhs_sep = [
            ex_kws_lhs + ex_kws_sep.join(kw) + ex_kws_rhs for kw in kws
        ]
    # txts
    if txts is not None:
        txts_w_lhs_rhs = [ex_txts_lhs + txt + ex_txts_rhs for txt in txts]
    # Create list of examples - both kws & txts
    if (kws is not None) and (txts is not None):
        examples_list = []
        for i in range(len(txts)):
            examples_list.append(kws_w_lhs_rhs_sep[i] + txts_w_lhs_rhs[i])
        examples_list.append(kws_w_lhs_rhs_sep[-1] + ex_txts_lhs)
    # Create list of examples - only kws
    elif (kws is not None) and (txts is None):
        examples_list = kws_w_lhs_rhs_sep
        examples_list.append(ex_kws_lhs)
    # Create list of examples - only txts
    elif (txts is not None) and (kws is None):
        examples_list = txts_w_lhs_rhs
        examples_list.append(ex_txts_lhs)
    # Create list of examples - None (empty list)
    elif (kws is None) and (txts is None):
        examples_list = []  # Numerals
    if ex_add_numeral and len(examples_list) != 0:
        for i in range(len(examples_list)):
            examples_list[i] = (
                str(i + 1) + ex_postnum_starter + examples_list[i]
            )
    # Assemblage
    examples = ex_sep.join(examples_list)
    prompt = intro + examples
    # Remove trailing right space
    if remove_right_trailing_space:
        prompt = prompt.rstrip(" ")
    return prompt


def make_prompt_w_keywords_new(
    intro: Optional[str] = "",
    kws: Optional[List[List[str]]] = None,
    txts: Optional[List[str]] = None,
    keywords_new: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Make prompt with kws and txts, ,using additional keywords

    This function is a simple wrapper around `prompt_formatter`, which
    essentially does kws += keywords_new, plus sanity checks,
    before passing these all to `prompt_formatter`.

    See the documentation for `prompt_formatter` for more.
    """
    if (kws is None) != (keywords_new is None):
        raise ValueError(
            "Either kws and keywords_new are both specified,"
            " or both are set to None"
        )
    if kws is not None:
        kws = copy.deepcopy(kws)
        kws.append(keywords_new)
    prompt = prompt_formatter(
        intro=intro,
        kws=kws,
        txts=txts,
        **kwargs,
    )
    return prompt


def trim_gen_seq(seq: str, prompt: str, end_delimiter: str) -> str:
    """
    Extract generated sequence

    Two steps:
    1. Remove `prompt` (replaced by "")
    2. Drop everything on the right of `end_delimiter`
    """
    seq = _left_trim_gen_seq(seq=seq, prompt=prompt)
    seq = _right_trim_gen_seq(seq=seq, end_delimiter=end_delimiter)
    return seq


def _left_trim_gen_seq(seq: str, prompt: str) -> str:
    """
    Drop the prompt from the generated sequence.
    """
    seq = seq.replace(prompt, "")
    return seq


def _right_trim_gen_seq(seq: str, end_delimiter: str):
    """
    Remove anything on the rhs of `end_delimiter`
    """
    seq = seq.split(end_delimiter)[0]
    return seq
