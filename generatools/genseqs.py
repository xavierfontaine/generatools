"""
Sequence generation
"""
import torch
import transformers
import logging
from typing import List, Optional


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
