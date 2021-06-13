import unittest
import pytest
from generatools.preproc import prompt_formatter, make_prompt_w_keywords_new


class TestPromptFormatter(unittest.TestCase):
    def test_only_intro_works(self):
        obs_out = prompt_formatter(intro="Hey lad!")
        exp_out = "Hey lad!"
        assert exp_out == obs_out

    def test_only_kws_works(self):
        obs_out = prompt_formatter(
            kws=[["hey", "lad"], ["howdy"]],
            ex_sep="[exsep]",
            ex_kws_sep="[kwsep]",
            ex_kws_lhs="[kwlhs]",
            ex_kws_rhs="[kwrhs]",
        )
        exp_out = (
            "[kwlhs]hey[kwsep]lad[kwrhs]"
            "[exsep]"
            "[kwlhs]howdy[kwrhs]"
            "[exsep]"
            "[kwlhs]"
        )
        assert exp_out == obs_out

    def test_only_txt_works(self):
        obs_out = prompt_formatter(
            txts=["I'm a cow", "mooh"],
            ex_sep="[exsep]",
            ex_txts_lhs="[txtlhs]",
            ex_txts_rhs="[txtrhs]",
        )
        exp_out = (
            "[txtlhs]I'm a cow[txtrhs]"
            "[exsep]"
            "[txtlhs]mooh[txtrhs]"
            "[exsep]"
            "[txtlhs]"
        )
        assert exp_out == obs_out

    def test_all_wo_numerals_work(self):
        obs_out = prompt_formatter(
            intro="This is an intro!",
            txts=["I'm a cow", "mooh"],
            kws=[
                ["hey", "lad"],
                ["howdy"],
                ["how"],
            ],
            ex_sep="[exsep]",
            ex_add_numeral=False,
            ex_kws_sep="[kwsep]",
            ex_kws_lhs="[kwlhs]",
            ex_kws_rhs="[kwrhs]",
            ex_txts_lhs="[txtlhs]",
            ex_txts_rhs="[txtrhs]",
        )
        exp_out = (
            "This is an intro!"
            "[kwlhs]hey[kwsep]lad[kwrhs][txtlhs]I'm a cow[txtrhs]"
            "[exsep]"
            "[kwlhs]howdy[kwrhs][txtlhs]mooh[txtrhs]"
            "[exsep]"
            "[kwlhs]how[kwrhs][txtlhs]"
        )
        assert exp_out == obs_out

    def test_all_w_numerals_work(self):
        obs_out = prompt_formatter(
            intro="This is an intro!",
            txts=["I'm a cow", "mooh"],
            kws=[
                ["hey", "lad"],
                ["howdy"],
                ["how"],
            ],
            ex_sep="[exsep]",
            ex_add_numeral=True,
            ex_postnum_starter="[postnum]",
            ex_kws_sep="[kwsep]",
            ex_kws_lhs="[kwlhs]",
            ex_kws_rhs="[kwrhs]",
            ex_txts_lhs="[txtlhs]",
            ex_txts_rhs="[txtrhs]",
        )
        exp_out = (
            "This is an intro!"
            "1[postnum][kwlhs]hey[kwsep]lad[kwrhs][txtlhs]I'm a cow[txtrhs]"
            "[exsep]"
            "2[postnum][kwlhs]howdy[kwrhs][txtlhs]mooh[txtrhs]"
            "[exsep]"
            "3[postnum][kwlhs]how[kwrhs][txtlhs]"
        )
        assert exp_out == obs_out

    def test_sanity_on_kws_and_txt_works(self):
        with pytest.raises(ValueError):
            prompt_formatter(
                txts=["I'm a cow", "mooh"],
                kws=[["hey", "lad"], ["howdy"]],
            )
        with pytest.raises(ValueError):
            prompt_formatter(
                txts=["I'm a cow", "mooh"],
                kws=[["hey", "lad"]],
            )
        with pytest.raises(ValueError):
            prompt_formatter(
                txts=["I'm a cow", "mooh", "moooooh"],
                kws=[["hey", "lad"], ["howdy"]],
            )

    def test_remove_trailing_right_space(self):
        obs_out = prompt_formatter(
            intro="This is an intro!   ", remove_right_trailing_space=True
        )
        exp_out = "This is an intro!"
        assert exp_out == obs_out


def test_make_prompt_w_keywords_new():
    obs_out = make_prompt_w_keywords_new(
        kws=[["hey "]],
        txts=["Jude"],
        keywords_new=["don't ", "be"],
        ex_sep="\n",
    )
    exp_out = "hey Jude\ndon't be"
    assert exp_out == obs_out
