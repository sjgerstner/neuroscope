"""single function recompute_acts"""
from os.path import exists
import pickle
from tqdm import tqdm

import torch
import einops

from datasets import Dataset

from utils import ModelWrapper, detect_cases, get_act_type_keys, adapt_activations
from b_activations import _get_reduce

def _recompute_from_cache(
    model:ModelWrapper, layer:int, neuron:int, indices_within_dataset:torch.Tensor, save_path:str, key:tuple[str,str,str]
) -> tuple[dict[str,torch.Tensor], torch.Tensor]:
    with open(f"{save_path}/activation_cache/batch_size.txt", 'r', encoding='utf-8') as f:
        batch_size = int(f.read())
    ln_cache=[]
    positions=[]
    for index_within_dataset in indices_within_dataset:
        index_within_dataset=int(index_within_dataset)#just in case
        batch_index, index_within_batch = divmod(index_within_dataset, batch_size)
        batch_file = f"{save_path}/activation_cache/batch{batch_index}"
        if exists(f"{batch_file}.pt"):
            saved_stuff = torch.load(f"{batch_file}.pt")
        else:
            assert exists(f"{batch_file}.pickle")
            with open(f"{batch_file}.pickle", 'rb') as f:
                saved_stuff = pickle.load(f)
        #ln cache
        subcache = saved_stuff['ln_cache'][index_within_batch] #(batch) pos layer d_model
        ln_cache.append(subcache[...,layer,:])
        #positions of max/min activations within sequence
        single_pos = saved_stuff[key]['indices'][index_within_batch,layer,neuron]
        positions.append(single_pos)
    ln_cache = torch.stack(ln_cache).cuda()
    positions = torch.stack(positions).cuda()

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
    return intermediate, positions

def _recompute_from_scratch(
    model:ModelWrapper, layer:int, neuron:int, indices_within_dataset:torch.Tensor, dataset:Dataset,
) -> dict[str,torch.Tensor]:
    input_ids = torch.stack([
        dataset[int(index_within_dataset)]['input_ids']
        for index_within_dataset in indices_within_dataset
    ])
    attention_mask = torch.stack([
        dataset[int(index_within_dataset)]['attention_mask']
        for index_within_dataset in indices_within_dataset
    ])
    names_filter = [
        f"blocks.{layer}.{hook}"
        for hook in ['hook_pre', 'hook_pre_linear', 'hook_post']
    ]
    raw_cache = model.run_with_cache(
        input_ids,
        attention_mask=attention_mask,
        names_filter=names_filter,
        return_type=None,
        stop_at_layer=layer+1,
    )
    intermediate = {
        hook: raw_cache[f"blocks.{layer}.{hook}"][...,neuron]
        for hook in ['hook_pre', 'hook_pre_linear', 'hook_post']
    }
    intermediate['swish'] = model.actfn(intermediate['hook_pre'])

    return intermediate

def recompute_acts(
    model:ModelWrapper,
    layer:int, neuron:int,
    dataset:Dataset,
    indices_within_dataset:torch.Tensor,
    save_path:str,
    key:tuple[str,str,str],
    use_cache:bool=True,
    ) -> dict[str,torch.Tensor|list[str]]:
    """Recompute activations for the given neuron and dataset indices, using cached residual stream activations.

    Args:
        model (ModelWrapper): the model
        layer (int): layer index
        neuron (int): neuron index within layer
        indices_within_dataset (torch.Tensor[int]): indices of relevant text examples within the dataset
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
    if use_cache:
        intermediate, positions = _recompute_from_cache(
            model=model,
            layer=layer,
            neuron=neuron,
            indices_within_dataset=indices_within_dataset,
            save_path=save_path,
            key=key,
        )
    else:
        intermediate = _recompute_from_scratch(
            model=model,
            layer=layer,
            neuron=neuron,
            indices_within_dataset=indices_within_dataset,
            dataset=dataset,
        )
    bins = detect_cases(
        gate_values=intermediate['hook_pre'],
        in_values=intermediate['hook_pre_linear'],
        keys=[key[0]],
        to_device='cuda',
    )
    for atk in act_type_keys:
        if atk not in intermediate:
            intermediate[atk] = bins[key[0]] * intermediate['_'.join(atk.split('_')[2:])]
            if torch.all(intermediate[atk]<=0):
                intermediate[atk][intermediate[atk]==-0.0]=-1e-7

    recomputed_acts = torch.stack([intermediate[hook] for hook in act_type_keys], dim=-1)

    if not use_cache:
        positions = torch.argmax(torch.abs(intermediate[f"{key[0]}_{key[1]}"]), dim=1)

    return {'all_acts':recomputed_acts, 'position_indices':positions, 'act_type_keys':act_type_keys}

def recompute_acts_if_necessary(args, summary_dict, maxmin_keys, neuron_dir, single_sign_to_adapt=1, **kwargs):
    activations_file = f'{neuron_dir}/activations{"_refactored" if args.refactor_glu else ""}.pt'
    activations_file_raw = f'{neuron_dir}/activations.pt'
    if exists(activations_file):
    #TODO we may need to comment this out because the internal format changed
        activation_data = torch.load(activations_file)
    elif args.refactor_glu and exists(activations_file_raw):
        activation_data = torch.load(activations_file_raw)
        if single_sign_to_adapt==-1:
            activation_data = adapt_activations(activation_data)
        torch.save(activation_data, activations_file)
    # if False:
    #     pass
    else:
        activation_data = {case_key:recompute_acts(
                **kwargs,
                key=case_key,
                indices_within_dataset=summary_dict[case_key]['indices'][...,kwargs['layer'],kwargs['neuron']],
                use_cache=args.use_cache,
            )
            for case_key in tqdm(maxmin_keys)}
        torch.save(activation_data, activations_file)
    return activation_data

def expand_with_summary(activation_data, summary_dict, layer, neuron):
    for key,value in summary_dict.items():
        if isinstance(value, torch.Tensor):
            activation_data[key]=value[...,layer,neuron]
        elif isinstance(value, dict):
            for key1,value1 in value.items():
                activation_data[key][key1]=value1[...,layer,neuron]
    return activation_data
