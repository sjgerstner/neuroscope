from torch import is_tensor

from datasets import Dataset
from transformers.activations import ACT2FN

from transformer_lens import HookedTransformer

class ModelWrapper(HookedTransformer):
    """Allows to directly access the (sub) activation function of the model,
    (i.e., Swish in the case of SwiGLU etc.)
    without looking it up every time
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actfn = ACT2FN[self.cfg.act_fn]

class DatasetWrapper(Dataset):
    """Allows to directly access the following properties of a dataset:
        self.n_tokens: total number of tokens
        self.max_seq_len: maximum length in tokens of a single example
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tokens = sum(len(row) for row in self['input_ids'])
        self.max_seq_len = max(len(row) for row in self['input_ids'])

def _move_to(dict_of_tensors, device):
    for key,value in dict_of_tensors.items():
        if is_tensor(value):
            dict_of_tensors[key] = value.to(device)
        elif isinstance(value, dict):
            dict_of_tensors[key] = _move_to(value, device)
    return dict_of_tensors
