"""single function recompute_acts"""
import pickle

import torch
import einops

#from transformers.activations import ACT2FN

from utils import ModelWrapper

def recompute_acts(model:ModelWrapper, layer:int, neuron:int, indices:torch.Tensor, save_path:str):
    """Recompute activations for the given neuron and dataset indices, using cached residual stream activations.

    Args:
        model (ModelWrapper): the model
        layer (int): layer index
        neuron (int): neuron index within layer
        indices (torch.Tensor[int]): indices of relevant text examples within the dataset
        save_path (str): path where cached residual stream activations are stored.
            Specifically, this should be the parent directory of "activation_cache".

    Returns:
        dict: dictionary with keys 'gate','in','acts', containing tensors of shape (sample, pos)
    """
    ln_cache=[]
    for i in indices:
        i=int(i)#just in case
        with open(f"{save_path}/activation_cache/batch{i}.pickle", 'rb') as f:
            subcache = pickle.load(f)['ln_cache'] #batch pos layer d_model
            ln_cache.append(subcache[:,:,layer,:])
            #print(ln_cache[-1].shape)
    ln_cache = torch.cat(ln_cache).cuda()
    #print(ln_cache.shape) #should be sample pos d_model

    intermediate = {}

    intermediate['gate'] = model.actfn(
        einops.einsum(ln_cache,
                      model.blocks[layer].mlp.W_gate[:,neuron],
                      'sample pos d_model, d_model -> sample pos'
                      )
        )
    intermediate['in'] = einops.einsum(
        ln_cache,
        model.blocks[layer].mlp.W_in[:,neuron],
        'sample pos d_model, d_model -> sample pos'
        )
    intermediate['acts'] = intermediate['gate']*intermediate['in']
    #TODO get position of max/min activation in sequence

    return intermediate
