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


def make_prompt_from_examples(
    intro: Optional[str] = None,
    examples: Optional[List[List]] = None,
    keywords_new: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Construct a prompt for generation

    Make a prompt from an intro, example (keywords, sentence) pairs and a
    new set of keywords. This function is a simple wrapper around
    `prompt_formatter`. It constructs `kws` and `txts`, and
    pass them to that function together with `intro` and `kwargs`.
    See docstring for `prompt_formater` for more on the purpose and output.

    `examples` and `keywords_new` (which replace kws and txts) are as follow:
    - `examples` is a list. Each example is a [[keywords1, keywords2, ...], txt]
      pair. The keywords list is left empty if none. The text should be set to
      None if none.
    - `keywords_new` is a tuple [keyword1, keyword2, ...] of additionnal
      keywords with no associated text, and will be append at the end of `kws`.

    Other arguments are the same as in `prompt_formatter`

    This construction allows for a more visual association between keywords and
    text wrt to `prompt_formatter`.

    Parameters
    ----------
    intro : Optional[str]
        intro
    examples : Optional[List[List]]
        examples
    keywords_new : Optional[List[str]]
        keywords_new
    kwargs :
        kwargs

    Returns
    -------
    str
    """
    # Sanity
    _check_make_prompt_from_examples_args_sanity(
        examples=examples, keywords_new=keywords_new
    )
    # Get keywords
    kws = [kw for kw, sent in examples]
    examples_include_keywords = any([len(k) > 0 for k in kws])
    if not examples_include_keywords:
        kws = None
    if _new_keywords_provided:
        kws.append(keywords_new)
    # Get sentences
    txts = [sent for kw, sent in examples]
    examples_include_sents = any([len(t) > 0 for t in txts])
    if not examples_include_sents:
        txts = None
    # Sanity checks on keywords
    if (keywords_new is not None) != examples_include_keywords:
        raise ValueError(
            "There should be either keywords through both `examples` and"
            " `keywords_new` or no keywords at all."
        )
    # Getting the prompt
    prompt = prompt_formatter(
        intro=intro,
        kws=kws,
        txts=txts,
        **kwargs,
    )
    return prompt


def _new_keywords_provided(keywords_new: Optional[List]) -> bool:
    """Consider keywords_new is provided if keywords_new is neither None nor an empty list"""
    if keywords_new is None:
        return False
    elif isinstance(keywords_new, list):
        if len(keywords_new) == 0:
            return False
    else:
        return True


def _check_make_prompt_from_examples_args_sanity(
    keywords_new: Optional[List],
    examples: Optional[List[List]],
) -> None:
    """Check keywords_new is either None or a list of strings"""
    # keywords_new should be None or list
    if (keywords_new is not None) and (not isinstance(keywords_new, list)):
        raise ValueError("keywords_new should be list or None")
    # If keywords_new is not empty, should be list of str
    if isinstance(keywords_new, list):
        if len(keywords_new) != 0:
            if not all([isinstance(x, str) for x in keywords_new]):
                raise ValueError("Elements of keywords_new should be str")
    # Either keywords in examples and keywords_new are given, or none of them
    # are given
    kws = [kw for kw, sent in examples]
    examples_include_keywords = any([len(k) > 0 for k in kws])
    if (keywords_new is not None) != examples_include_keywords:
        raise ValueError(
            "There should be either keywords through both `examples` and"
            " `keywords_new` or no keywords at all."
        )
    # Each example should be of length 2
    all_examples_are_length_2 = all([len(e) == 2 for e in examples])
    if not all_examples_are_length_2:
        raise ValueError(
            "All examples should be lists of length 2. So docstring on writing"
            " examples without keywords or text."
        )
