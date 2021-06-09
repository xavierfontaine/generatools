import torch
import transformers as trf
import tqdm
import logging
from typing import List, Union

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
