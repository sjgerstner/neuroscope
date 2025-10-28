"""single function recompute_acts"""
import pickle

import torch
import einops

from utils import ModelWrapper, detect_cases, get_act_type_keys

def recompute_acts(
    model:ModelWrapper, layer:int, neuron:int, indices:torch.Tensor, save_path:str, key:tuple[str,str,str]
    ) -> dict[str,torch.Tensor|list[str]]:
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
    act_type_keys = get_act_type_keys(key)
    if not act_type_keys:
        return {}
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

    bins = detect_cases(
        gate_values=intermediate['hook_pre'],
        in_values=intermediate['hook_pre_linear'],
        keys=[key[0]]
    )
    for atk in act_type_keys:
        if atk not in intermediate:
            intermediate[atk] = bins[key[0]] * intermediate['_'.join(atk.split('_')[2:])]
            #hack: convert -0.0 to a small negative value
            if key[-1]=='min':
                intermediate[atk][intermediate[atk]==-0.0]=-1e-7

    recomputed_acts = torch.stack([intermediate[hook] for hook in act_type_keys], dim=-1)

    return {'all_acts':recomputed_acts, 'position_indices':positions, 'act_type_keys':act_type_keys}
