"""single function recompute_acts"""
import pickle

import torch
import einops

#from transformers.activations import ACT2FN

from utils import ModelWrapper, VALUES_TO_SUMMARISE

def recompute_acts(
    model:ModelWrapper, layer:int, neuron:int, indices:torch.Tensor, save_path:str, key:tuple[str,str,str]
    ) -> dict[str,torch.Tensor]:
    """Recompute activations for the given neuron and dataset indices, using cached residual stream activations.

    Args:
        model (ModelWrapper): the model
        layer (int): layer index
        neuron (int): neuron index within layer
        indices (torch.Tensor[int]): indices of relevant text examples within the dataset
        save_path (str): path where cached residual stream activations are stored.
            Specifically, this should be the parent directory of "activation_cache".

    Returns:
        torch.Tensor[float]:
            the first dimension corresponds to the different VALUES_TO_SUMMARISE,
            then we have batch and position
    """
    ln_cache=[]
    positions=[]
    for i in indices:
        i=int(i)#just in case
        with open(f"{save_path}/activation_cache/batch{i}.pickle", 'rb') as f:
            saved_stuff = pickle.load(f)
            #ln cache
            subcache = saved_stuff['ln_cache'] #batch pos layer d_model
            ln_cache.append(subcache[:,:,layer,:])
            #positions of max/min activations within sequence
            single_pos = saved_stuff[key]['indices'][:,layer,neuron]
            positions.append(single_pos)
    ln_cache = torch.cat(ln_cache).cuda()
    positions = torch.cat(positions).cuda()

    intermediate = {}
    intermediate['hook_pre'] = einops.einsum(
        ln_cache,
        model.blocks[layer].mlp.W_gate[:,neuron],
        'sample pos d_model, d_model -> sample pos'
    )
    intermediate['swish'] = model.actfn(intermediate['hook_pre'])
    intermediate['hook_pre_linear'] = einops.einsum(
        ln_cache,
        model.blocks[layer].mlp.W_in[:,neuron],
        'sample pos d_model, d_model -> sample pos'
    )
    intermediate['hook_post'] = intermediate['swish']*intermediate['hook_pre_linear']

    recomputed_acts = torch.stack([intermediate[hook] for hook in VALUES_TO_SUMMARISE], dim=-1)

    return {'all_acts':recomputed_acts, 'position_indices':positions}
