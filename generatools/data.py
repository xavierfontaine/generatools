import numpy as np
import torch
import transformers as trf
import tqdm
import logging
import copy
from typing import List, Union, Optional

import generatools.utils.logging as utils_logging


logger = logging.getLogger(__name__)


class GPTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: trf.tokenization_utils.PreTrainedTokenizerBase,
        concat_token: Union[str, None] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Union[int, None] = None,
        drop_too_long: bool = False,
        special_bos: Union[str, None] = None,
        special_eos: Union[str, None] = None,
    ) -> None:
        """Dataset class for causal GPT-like LM models

        Parameters
        ----------
        texts : List[str]
            texts tokenizer : transformers.tokenization_utils.PreTrainedTokenizerBase
            tokenizer
        sequence_length: Union[int, None]
            Length to be reached through padding or concatenation.
        padding: bool
            Should we pad? Default to False.
        truncation: bool
            Should be truncate? Default to True.
        concat_token: Union[str, None]
            None if no concatenation.
        max_length: Union[int, None]
            maximum sequence length. Used when padding or truncation is true,
            or when `concat_token` is specified.
        drop_too_long: bool,
            Should we drop sentences whose length are larger than max_length?
        special_bos : Union[str, None]
            special_bos
        special_eos : Union[str, None]
            special_eos
        """
        # Init
        preproc_texts = texts
        tokenizer_args = dict()
        # Sanity checks
        self._check_args(
            tokenizer=tokenizer,
            concat_token=concat_token,
            padding=padding,
            special_bos=special_bos,
            special_eos=special_eos,
            max_length=max_length,
            truncation=truncation,
            drop_too_long=drop_too_long,
        )
        # Preprocessing
        if special_bos:
            preproc_texts = [special_bos + text for text in preproc_texts]
        if special_eos:
            preproc_texts = [text + special_eos for text in preproc_texts]
        if (
            padding
            or truncation
            or drop_too_long
            or (concat_token is not None)
        ):
            tokenizer_args["max_length"] = max_length
        # Concatenator
        if concat_token is not None:
            raise NotImplementedError("Concatenation is yet to be implemented")
        # Padding: change tokenizer arg
        if padding:
            tokenizer_args["padding"] = "max_length"
        # Truncation: change tokenizer arg
        if truncation:
            tokenizer_args["truncation"] = True
        if drop_too_long:
            # We increase a bit the max_length used during tokenization, so
            # that we can find which sequence was too long afterward.
            # We also set "truncation" to True to save memory.
            # NOTE : keep this to max_length + 1, as this function will rely on
            # this specific length afterward.
            tokenizer_args["max_length"] = max_length + 1
            tokenizer_args["truncation"] = True
        # Tokenization
        tok_data = [
            tokenizer(text=text, **tokenizer_args)
            for text in tqdm.tqdm(preproc_texts, desc="Tokenization")
        ]
        # Drop sentences that are too long
        if drop_too_long:
            n_whole_data = len(
                tok_data
            )  #  Keeping track of original # of data
            #  Find datapoints that are too long (length equal to max_length+1
            #  and the last token is not and eos or padding token)
            tok_data_lens = [len(doc["input_ids"]) for doc in tok_data]
            assert max(tok_data_lens) <= max_length + 1  #  Enforced above
            too_long_idx = [
                i
                for i, toks in enumerate(tok_data)
                if (len(toks["input_ids"]) == max_length + 1)
                and toks["input_ids"][-1]
                not in [tokenizer.eos_token_id, tokenizer.pad_token_id]
            ]
            tok_data = [
                tok_data[i]
                for i in range(len(tok_data))
                if i not in too_long_idx
            ]
            # Truncate data to the right length
            for i in range(len(tok_data)):
                if len(tok_data[i]["input_ids"]) == max_length + 1:
                    del tok_data[i]["input_ids"][-1]
                    del tok_data[i]["attention_mask"][-1]
            assert (
                max([len(doc["input_ids"]) for doc in tok_data]) <= max_length
            )
            logger.info(
                f"Dropped {len(too_long_idx)}/{n_whole_data} docs for which num of"
                f" tokens was greater than {max_length}"
            )
        self.tok_data = tok_data

    def _check_args(
        self,
        tokenizer: trf.tokenization_utils.PreTrainedTokenizerBase,
        concat_token: Union[str, None],
        padding: bool,
        special_bos: Union[str, None],
        special_eos: Union[str, None],
        max_length: Union[int, None],
        truncation: bool,
        drop_too_long: bool,
    ) -> None:
        """Sanity checks on arguments"""
        if concat_token and padding:
            utils_logging.log_and_raise(
                ValueError, "Cannot both concatenate and pad"
            )
        if (
            (concat_token is not None)
            or padding
            or truncation
            or drop_too_long
        ) and not max_length:
            utils_logging.log_and_raise(
                ValueError,
                "max_length should be provided with concat_token or padding or"
                " truncation or drop_too_long",
            )
        if max_length and not (
            (concat_token is not None)
            or padding
            or truncation
            or drop_too_long
        ):
            logger.warning(
                "max_length is set without concat_token or padding or truncation.  Sequences won't be augmented, but will be truncated by default."
            )
        if drop_too_long and truncation:
            logger.warning(
                "Both 'drop_too_long' and 'truncation' have been set to True. "
                "Note that all sequences with length > max_length will be dropped, "
                "so truncation makes no sense here."
            )
        # Check required tokens are in the tokenizer
        if padding:
            if tokenizer.pad_token is None:
                utils_logging.log_and_raise(
                    ValueError,
                    "tokenizer.pad_token should be set when padding=True",
                )
        if special_bos:
            if tokenizer.bos_token is None:
                utils_logging.log_and_raise(
                    ValueError,
                    "tokenizer.pad_token should be set when special_bos not"
                    " None",
                )
        if special_eos:
            if tokenizer.eos_token is None:
                utils_logging.log_and_raise(
                    ValueError,
                    "tokenizer.pad_token should be set when special_eos not"
                    " None",
                )

    def __getitem__(self, idx):
        return self.tok_data[idx]

    def __len__(self):
        return len(self.tok_data)


# TODO : docstr
def gpttokenizer_collate_fn(dicts_list: List[dict]) -> dict:
    """
    Collate_fn for a list of outputs of GPT2Tokenizer.

    To be used as value for the collate_fn argument in torch.data.Datalaoder .
    """
    gpttok_dtypes = {
        "input_ids": torch.long,
        "labels": torch.long,
        "attention_mask": torch.float,
    }
    # Init
    keys = dicts_list[0].keys()
    out_dict = dict()
    # Transform into list of list dict
    for k in keys:
        out_dict[k] = [dic[k] for dic in dicts_list]
    # Create labels
    out_dict["labels"] = copy.deepcopy(out_dict["input_ids"])
    # Create tensors
    out_dict = {
        k: torch.Tensor(v).type(gpttok_dtypes[k]) for k, v in out_dict.items()
    }
    return out_dict


def splitter(
    dataset: torch.utils.data.Dataset,
    val_prop: int,
    test_prop: int,
    train_subsample_prop: float,
    seed: Optional[int],
) -> List[torch.utils.data.Dataset]:
    """Split dataset into train/test/val

    The val and test datasets are the same irrespective of the subsamplif of
    the train dataset through `train_subsample_prop`. This allows playing with
    the train test size without affecting val and test metrics.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset
    val_prop : int
        val_prop
    test_prop : int
        test_prop
    train_subsample_prop : float
        train_subsample_prop
    seed : int
        seed

    Returns
    -------
    List[torch.utils.data.Dataset]
        (train dataset, val dataset, test dataset)
    """
    dataset_n = len(dataset)
    # Get sizes
    val_size = round(val_prop * dataset_n)
    test_size = round(test_prop * dataset_n)
    train_size = dataset_n - (val_size + test_size)
    # Shuffle Ids
    ids = list(range(dataset_n))
    np.random.seed(seed)
    np.random.shuffle(ids)
    # Get ranges
    first_val_idx = train_size
    first_test_idx = first_val_idx + val_size
    # Subset
    dataset_train = dataset[:first_val_idx]
    dataset_val = dataset[first_val_idx:first_test_idx]
    dataset_test = dataset[first_test_idx:]
    # Logging
    logger.info(
        f"Split datasets into train ({len(dataset_train)} rows), val"
        f" ({len(dataset_val)}) and test ({len(dataset_test)})"
    )
    # Subsampling train dataset
    assert 0 < train_subsample_prop <= 1
    if train_subsample_prop < 1:
        previous_train_len = len(dataset_train)
        last_train_idx = round(len(dataset_train) * train_subsample_prop)
        dataset_train = dataset_train[:last_train_idx]
        logger.info(
            f"Subsetted the train dataset, keeping only {len(dataset_train)}/"
            f"{previous_train_len} rows ({int(train_subsample_prop*100)}%).)"
        )
    return dataset_train, dataset_val, dataset_test
