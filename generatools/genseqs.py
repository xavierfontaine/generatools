"""
Sequence generation
"""
import torch
import transformers
import logging
import mlflow
import copy
import os
from typing import List, Optional, Union

import generatools.utils.mlflow
import generatools.utils.transformers
import generatools.sequences
import generatools.preproc
import generatools.genseqs
import generatools.hyperopt


logger = logging.getLogger(__name__)


def gen_seqs_from_prompt(
    prompt: str,
    model: transformers.modeling_utils.PreTrainedModel,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizerBase,
    max_length_after_prompt: int,
    num_return_sequences: int,
    device: torch.device,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    """Generate `num_return_sequences` sequences based on `prompt`

    Max length for sequences is len(prompt) + max_length_after_prompt.
    Additionnal kwargs are passed to model.generate.
    """
    # Tokenization
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids_size = input_ids.shape[1]
    logger.info("-- Prompt of size {}.".format(input_ids_size))
    # Prediction
    transformers.trainer_utils.set_seed(seed)
    output = model.generate(
        input_ids,
        max_length=input_ids_size + max_length_after_prompt,
        return_dict_in_generate=False,
        output_scores=False,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        **kwargs
    )
    y_seqs = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return y_seqs


def run_grid_generation_from_conf(conf: dict, conf_filepath: str):
    """
    Run multiple generations based on a configuration file.

    See the configuration file template for more.
    """
    # Device
    if conf["use_gpu"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Set experiment path's global variable
    mlflow.set_tracking_uri(uri=conf["mlflow_expe_dirpath"])
    # Get id & set expe's global variable
    experiment_id = generatools.utils.mlflow.create_expe(
        expe_name=conf["mlflow_expe_name"],
    )
    mlflow.set_experiment(
        experiment_name=conf["mlflow_expe_name"],
    )
    # Getting param grid
    param_grid = generatools.hyperopt.generate_grid_from_conf(
        conf=conf["hyperparams"]
    )

    # Loop
    loaded_mdl_params = None  # params used during last model loading
    for params in param_grid:
        run_already_exist = generatools.utils.mlflow.run_w_params_exists(
            params=params
        )
        # If no run with those parameters, proceed
        if not run_already_exist:
            # Should we load tokenizer & model?
            load_tok_mdl = _check_should_load_tok_and_mdl(
                previous_params=loaded_mdl_params, params=params
            )
            # Load if necessary
            if load_tok_mdl:
                logger.info(
                    "Load model & tokenizer (not loaded for previous set of params)"
                )
                (
                    tokenizer,
                    model,
                ) = generatools.utils.transformers.load_tokenizer_model(
                    model_class=params["model_class"],
                    model_name=params["model_name"],
                    tokenizer_class=params["tokenizer_class"],
                    tokenizer_name=params["tokenizer_name"],
                    device=device,
                )
                loaded_mdl_params = copy.deepcopy(params)
            else:
                logger.info("Reusing previously loaded model & tokenizer")
            with mlflow.start_run(experiment_id=experiment_id):
                logger.info("Running expe with parameters {}".format(params))
                # prompt_seqs_pair_list: list of dict, une per keywords_new,
                # each element of the form
                # {keywords_new, [gen_seq_1, ,..., gen_seq_S], [gen_seq_trimmed_1, ...]}
                prompt_seqs_pair_list = []
                keywords_new_list = conf["keywords_new_list"]
                if keywords_new_list == [] or keywords_new_list == [[]]:
                    # If no new list of keywords specified, we will use "None"
                    keywords_new_list = [None]
                for keywords_new in keywords_new_list:
                    prompt = generatools.preproc.make_prompt_w_keywords_new(
                        intro=params["intro"],
                        kws=params["examples"]["kws"],
                        txts=params["examples"]["txts"],
                        keywords_new=keywords_new,
                        ex_sep=params["ex_sep"],
                        ex_add_numeral=params["ex_add_numeral"],
                        ex_postnum_starter=params["ex_postnum_starter"],
                        ex_kws_sep=params["ex_kws_sep"],
                        ex_kws_lhs=params["ex_kws_lhs"],
                        ex_kws_rhs=params["ex_kws_rhs"],
                        ex_txts_lhs=params["ex_txts_lhs"],
                        ex_txts_rhs=params["ex_txts_rhs"],
                        remove_right_trailing_space=params[
                            "remove_right_trailing_space"
                        ],
                    )
                    logger.info("Prompt: {}".format(prompt))
                    sequences = generatools.genseqs.gen_seqs_from_prompt(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        seed=conf["seed"],
                        max_length_after_prompt=conf[
                            "max_length_after_prompt"
                        ],
                        num_return_sequences=conf["n_seqs"],
                        # As kwargs
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        top_k=params["top_k"],
                        repetition_penalty=params["repetition_penalty"],
                    )
                    sequences_trimmed = [
                        generatools.preproc.trim_gen_seq(
                            seq=gen_seq,
                            prompt=prompt,
                            end_delimiter=params["genseq_end_delimiter"],
                        )
                        for gen_seq in sequences
                    ]
                    prompt_seqs_pair = generatools.sequences.PromptSeqsPair(
                        prompt=prompt,
                        keywords=keywords_new,
                        sequences=sequences,
                        sequences_trimmed=sequences_trimmed,
                    )
                    prompt_seqs_pair_list.append(prompt_seqs_pair)
                # Store parameters
                mlflow.log_params(params)
                # Store output dict as a json artifact
                dictified_prompt_seqs_pair_list = [  # Passing prompt_seqs_pair to dict for
                    # mlflow.log_dict
                    prompt_seqs_pair.to_dict()
                    for prompt_seqs_pair in prompt_seqs_pair_list
                ]
                mlflow.log_dict(
                    dictionary=dictified_prompt_seqs_pair_list,
                    artifact_file=conf["mlflow_results_json_name"],
                )
                # generatools.utils.mlflow.store_dict_to_json_artifact(
                #    dic=xnew_ygen_list, filename=conf["mlflow_results_json_name"]
                # )
                # Store configuration file
                mlflow.log_artifact(conf_filepath)
                # Store current script
                script_filepath = os.path.realpath(__file__)
                mlflow.log_artifact(script_filepath)
                # Store readme
                mlflow.log_text(
                    text=conf["readme"], artifact_file="README.txt"
                )
        # If this run has already be done, not need to proceed
        elif run_already_exist:
            logger.info(
                "NOT running expe, as already exists. Parameters were {}".format(
                    params
                )
            )
        logger.info("\n")


def _check_should_load_tok_and_mdl(
    params: dict, previous_params: Union[dict, None]
) -> bool:
    """Should we load the tokenizer and model in params?

    Sharing of model and tokenizer is check along these keys:
    "model_class", "model_name", "tokenizer_class", "tokenizer_name"


    Parameters
    ----------
    params : dict
        params
    previous_params : Union[dict, None]
        dict if any, None else

    Returns
    -------
    bool
    """
    if previous_params is None:
        should_load_tok_mdl = True
    elif previous_params is not None:
        if (
            params["model_class"],
            params["model_name"],
            params["tokenizer_class"],
            params["tokenizer_name"],
        ) != (
            previous_params["model_class"],
            previous_params["model_name"],
            previous_params["tokenizer_class"],
            previous_params["tokenizer_name"],
        ):
            should_load_tok_mdl = True
        else:
            should_load_tok_mdl = False
    return should_load_tok_mdl
