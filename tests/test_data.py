import pytest
import transformers as trf
import unittest
import torch.utils.data

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


class TestDataLoader(unittest.TestCase):
    """
    Integration of data.GPTDataset + data.gpttokenizer_collate_fn in pytorch
    dataloader
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
        self.max_len = 11
        dataset = data.GPTDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_len,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=3,
            num_workers=0,
            pin_memory=True,
            collate_fn=data.gpttokenizer_collate_fn,
            drop_last=True,
        )
        self.dataloader_out = dataloader.__iter__().next()

    @pytest.mark.slow
    def test_integration_dataloader(self):
        self.correct_keys()
        self.correct_sizes()

    def correct_keys(self):
        assert self.dataloader_out.keys() == set(
            ["input_ids", "attention_mask", "labels"]
        )

    def correct_sizes(self):
        assert self.dataloader_out["input_ids"].shape == (
            len(self.texts),
            self.max_len,
        )


class DummyDataset(torch.utils.data.Dataset):
    """
    For testing data.splitter
    """

    def __init__(self):
        self.data = [[1, 2], [3], [4, 5, 6], [7]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestSplitter(unittest.TestCase):
    def setUp(self):
        # Dummy dataset class
        self.dataset = DummyDataset()

    def test_stability(self):
        splits_1 = data.splitter(
            dataset=self.dataset,
            val_prop=0.2,
            test_prop=0.2,
            train_subsample_prop=1,
            seed=1,
        )
        splits_2 = data.splitter(
            dataset=self.dataset,
            val_prop=0.2,
            test_prop=0.2,
            train_subsample_prop=1,
            seed=1,
        )
        assert splits_1 == splits_2

    def test_subsample_train_works(self):
        splits_subsamble = data.splitter(
            dataset=self.dataset,
            val_prop=0.2,
            test_prop=0.2,
            train_subsample_prop=1 / 3,
            seed=1,
        )
        assert len(splits_subsamble[0]) == 1

    def test_val_test_stable_when_subsample_train(self):
        splits_1 = data.splitter(
            dataset=self.dataset,
            val_prop=0.2,
            test_prop=0.2,
            train_subsample_prop=1,
            seed=1,
        )
        splits_subsamble = data.splitter(
            dataset=self.dataset,
            val_prop=0.2,
            test_prop=0.2,
            train_subsample_prop=1 / 3,
            seed=1,
        )
        assert splits_1[1:] == splits_subsamble[1:]
