import transformers
import torch
import gc


def load_tokenizer_model(
    model_class: str,
    model_name: str,
    tokenizer_class: str,
    tokenizer_name: str,
    device: torch.device,
) -> (
    transformers.tokenization_utils.PreTrainedTokenizerBase,
    transformers.modeling_utils.PreTrainedModel,
):
    """Programmatically load tokenizer & model from pretrained.

    Parameters
    ----------
    model_class : str
        Name of class as attached to the `transformer module
    model_name : str
        Name to be used in `from_pretrained`
    tokenizer_class : str
        Name of class as attached to the `transformer module
    tokenizer_name : str
        Name to be used in `from_pretrained`
    """
    tokenizer = getattr(transformers, tokenizer_class).from_pretrained(
        tokenizer_name
    )
    model = getattr(transformers, model_class).from_pretrained(model_name)
    gc.collect()
    model = model.to(device)
    return tokenizer, model
