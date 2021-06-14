import unittest
import generatools.sequences as sequences


class TestPromptSeqsPairSequences(unittest.TestCase):
    """
    Test that the x_sequences class behave properly.

    Test type: functional (public API)
    """

    def test_run_if_proper_sequences_trimmed(self):
        sequences.PromptSeqsPair(
            prompt="bla", sequences=["aa", "bb"], sequences_trimmed=["a", "b"]
        )

    def test_except_if_improper_sequences_trimmed(self):
        self.assertRaises(
            ValueError,
            sequences.PromptSeqsPair,
            prompt="bla",
            sequences=["aa", "bb"],
            sequences_trimmed=["a"],
        )

    def test_run_if_proper_prompt_lvl_eval(self):
        z = sequences.PromptSeqsPair(
            prompt="bla",
            sequences=["aa", "bb"],
            prompt_lvl_eval={"metric1": 2, "metric2": 3},
        )
        z.prompt = "bla"

    def test_except_if_improper_prompt_lvl_eval(self):
        self.assertRaises(
            TypeError,
            sequences.PromptSeqsPair,
            prompt="bla",
            sequences=["aa", "bb"],
            prompt_lvl_eval=[2, 3],
        )
        self.assertRaises(
            TypeError,
            sequences.PromptSeqsPair,
            prompt="bla",
            sequences=["aa", "bb"],
            prompt_lvl_eval={"metric1": [2, 3]},
        )

    def test_run_if_proper_seq_lvl_eval(self):
        sequences.PromptSeqsPair(
            prompt="bla",
            sequences=["aa", "bb"],
            seq_lvl_eval={"metric1": [2, 3], "metric2": [3, 4]},
        )

    def test_except_if_improper_seq_lvl_eval(self):
        self.assertRaises(
            TypeError,
            sequences.PromptSeqsPair,
            prompt="bla",
            sequences=["aa", "bb"],
            seq_lvl_eval=[
                {"metric1": 2, "metric2": 3},
                {"metric1": 2, "metric2": 3},
            ],
        )
        self.assertRaises(
            ValueError,
            sequences.PromptSeqsPair,
            prompt="bla",
            sequences=["aa", "bb"],
            seq_lvl_eval={"metric1": [2, 3], "metric2": [3]},
        )

    def test_to_dict_generates_a_dict(self):
        obs_out = sequences.PromptSeqsPair(
            prompt="bla",
            sequences=["aa", "bb"],
        ).to_dict()
        self.assertIsInstance(obs_out, dict)

    def test_average_seq_lvl_eval_works(self):
        exp_out = {"m1": 2, "m2": 20}
        obs_out = sequences.PromptSeqsPair(
            prompt="bla",
            sequences=["aa", "bb"],
            seq_lvl_eval={"m1": [1, 3], "m2": [10, 30]},
        ).average_seq_lvl_eval()
        self.assertEqual(exp_out, obs_out)


class TestPromptSeqsPairsList(unittest.TestCase):
    def test_setting_retrieving_works(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 1, "m2": 10},
            ),
            sequences.PromptSeqsPair(
                prompt="b",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 3, "m2": 30},
            ),
        ]
        pair_list = sequences.PromptSeqsPairsList(ls=ls)
        self.assertEqual(pair_list[1].prompt, "b")

    def test_fail_if_wrong_ls_input(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 1, "m2": 10},
            ),
            "oups",
        ]
        self.assertRaises(ValueError, sequences.PromptSeqsPairsList, ls=ls)

    def test_raise_exception_when_different_metric_set(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"oups": 1, "m2": 10},
            ),
            sequences.PromptSeqsPair(
                prompt="b",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 3, "m2": 30},
            ),
        ]
        pair_list = sequences.PromptSeqsPairsList(ls=ls)
        self.assertRaises(KeyError, pair_list.average_prompt_lvl_metrics)

    def test_prompt_lvl_averaging_works(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 1, "m2": 10},
            ),
            sequences.PromptSeqsPair(
                prompt="b",
                sequences=["y1", "y2"],
                prompt_lvl_eval={"m1": 3, "m2": 30},
            ),
        ]
        pair_list = sequences.PromptSeqsPairsList(ls=ls)
        exp_out = {"m1": 2, "m2": 20}
        obs_out = pair_list.average_prompt_lvl_metrics()
        self.assertEqual(exp_out, obs_out)

    def test_seq_lvl_averaging_works(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                seq_lvl_eval={"m1": [1, 1], "m2": [10, 10]},
            ),
            sequences.PromptSeqsPair(
                prompt="b",
                sequences=["y1", "y2"],
                seq_lvl_eval={"m1": [3, 3], "m2": [30, 30]},
            ),
        ]
        pair_list = sequences.PromptSeqsPairsList(ls=ls)
        exp_out = {"m1": 2, "m2": 20}
        obs_out = pair_list.average_seq_lvl_metrics()
        self.assertEqual(exp_out, obs_out)

    def test_to_json_works(self):
        ls = [
            sequences.PromptSeqsPair(
                prompt="a",
                sequences=["y1", "y2"],
                seq_lvl_eval={"m1": [1, 1], "m2": [10, 10]},
            ),
            sequences.PromptSeqsPair(
                prompt="b",
                sequences=["y1", "y2"],
                seq_lvl_eval={"m1": [3, 1], "m2": [30, 10]},
            ),
        ]
        pair_list = sequences.PromptSeqsPairsList(ls=ls)
        obs_out = pair_list.to_json()
        exp_out = [ls[0].to_dict(), ls[1].to_dict()]
        self.assertEqual(exp_out, obs_out)
