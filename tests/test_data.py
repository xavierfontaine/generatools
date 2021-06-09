import pytest
import transformers as trf
import unittest

from generatools import data


class TestGPTDataset(unittest.TestCase):
    """
    Test the torch.utils.data.Dataset child for GPT-like data.
    """

    def setUp(self):
        self.tokenizer = trf.GPT2Tokenizer.from_pretrained(
            "distilgpt2",
            pad_token="<|pad|>",
            special_bos="<|spec_bos|>",
            special_eos="<|spec_eos|>",
        )
        self.texts = [
            "I am Will Smith",
            "I love cow boys and I'm Prince of Bel Air",
            "Hey",
        ]

    @pytest.mark.slow
    def test_all(self):
        """
        Grouping tests to avoid loading the tokenizer more than once
        """
        self.correct_length()
        self.correct_item_format()
        self.vanilla_output()
        self.bos_eos_added()
        self.padding_effective()
        self.truncation_effective()
        self.truncation_and_padding_effective()
        self.drop_too_long_works()

    def correct_length(self):
        dataset = data.GPTDataset(texts=self.texts, tokenizer=self.tokenizer)
        assert len(dataset) == 3

    def correct_item_format(self):
        dataset = data.GPTDataset(texts=self.texts, tokenizer=self.tokenizer)
        assert dataset[0].keys() == set(["input_ids", "attention_mask"])

    def vanilla_output(self):
        dataset = data.GPTDataset(texts=self.texts, tokenizer=self.tokenizer)
        exp_lens = [4, 11, 1]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens

    def bos_eos_added(self):
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            special_bos=self.tokenizer.bos_token,
            special_eos=self.tokenizer.eos_token,
        )
        # Check lengths
        exp_lens = [6, 13, 3]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens
        #  Check first and last tokens
        assert dataset[0]["input_ids"][0] == self.tokenizer.bos_token_id
        assert dataset[0]["input_ids"][-1] == self.tokenizer.eos_token_id

    def padding_effective(self):
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            padding=True,
            max_length=15,
        )
        exp_lens = [15, 15, 15]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens
        assert dataset[0]["input_ids"][-1] == self.tokenizer.pad_token_id

    def truncation_effective(self):
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
        )
        exp_lens = [4, 10, 1]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens

    def truncation_and_padding_effective(self):
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=10,
        )
        exp_lens = [10, 10, 10]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens

    def drop_too_long_works(self):
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            drop_too_long=True,
            max_length=10,
        )
        exp_lens = [4, 1]
        obs_lens = [len(d["input_ids"]) for d in dataset]
        assert exp_lens == obs_lens
