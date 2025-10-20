"""
Compute max activations for all neurons on a given dataset.
The code was written with batch size 1 in mind.
Other batch sizes will almost certainly lead to bugs.
"""

#TODO (also other files) pathlib
#TODO replace pickle with pt
#TODO enable other batch sizes, don't forget problems with padding tokens

from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
import einops

from transformer_lens import HookedTransformer

import datasets

from utils import ModelWrapper, _move_to, add_properties

HOOKS_TO_CACHE = ['ln2.hook_normalized', 'mlp.hook_post', 'mlp.hook_pre', 'mlp.hook_pre_linear']
VALUES_TO_SUMMARISE = ['hook_post', 'hook_pre_linear', 'hook_pre', 'swish']
CASES = ['gate+_in+',
        'gate+_in-',
        'gate-_in+',
        'gate-_in-']
REDUCTIONS = ['max', 'min', 'sum']

def _get_reduce_and_arg(cache_item, reduction, k=1, to_device='cpu'):
    if reduction not in ('max', 'min', 'top', 'bottom'):
        raise NotImplementedError(f"reduction {reduction} not implemented")

    #... ? layer neuron -> ... k layer neuron
    myred = torch.topk(
        cache_item,
        dim=-3,
        k=k,
        largest=reduction in ('max','top'),
    )
    vi_dict = {
        'values': myred.values.to(to_device),
        'indices':myred.indices.to(dtype=torch.int, device=to_device),
    }
    if k==1:
        for key,tensor in vi_dict.items():
            vi_dict[key] = einops.reduce(
                tensor, '... 1 layer neuron -> ... layer neuron', reduction
            )
    return vi_dict

def _get_reduce(cache_item, reduction, arg=False, k=1, use_cuda=True, to_device='cpu'):
    if use_cuda and torch.cuda.is_available():
        cache_item = cache_item.cuda()

    if arg:
        return _get_reduce_and_arg(cache_item, reduction, k=k, to_device=to_device)
    return einops.reduce(
            cache_item,
            '... layer neuron -> layer neuron',
            reduction
            ).to(to_device)

def _get_all_neuron_acts(model, ids_and_mask, names_filter, max_seq_len=1024):
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb

    intermediate = {}

    batch_size = ids_and_mask['input_ids'].shape[0]
    seq_len = ids_and_mask['input_ids'].shape[1]

    _logits, raw_cache = model.run_with_cache(
        ids_and_mask['input_ids'],
        attention_mask=ids_and_mask['attention_mask'],
        names_filter=names_filter
        )
    #ActivationCache
    # with keys 'blocks.layer.mlp.hook_post' etc
    # and entries mostly with shape (batch pos neuron)

    mask = einops.rearrange(ids_and_mask['attention_mask'], 'batch pos -> batch pos 1 1').cuda()
    #batch pos neuron
    cache={}
    for key_to_summarise in HOOKS_TO_CACHE:
        cache[key_to_summarise] = torch.stack(
            [raw_cache[f'blocks.{layer}.{key_to_summarise}'] for layer in range(model.cfg.n_layers)],
            dim=-2,#batch pos neuron/d_model -> batch pos layer neuron/d_model
        )
        cache[key_to_summarise] *= mask
    del raw_cache

    #ln_cache: initialise with zeros (batch pos layer d_model)
    intermediate['ln_cache'] = torch.zeros(
        (batch_size, max_seq_len, model.cfg.n_layers, model.cfg.d_model)
        )
    #fill in
    intermediate['ln_cache'][:, :seq_len, :] = cache['ln2.hook_normalized'].cpu()

    #summary keys (mean and frequencies)
    #layer neuron
    #intermediate['mean'] = _get_reduce(cache['mlp.hook_post'], 'sum')#not needed anymore
    gate_positive = cache['mlp.hook_pre']>0
    in_positive = cache['mlp.hook_pre_linear']>0
    bins={}
    bins['gate+_in+'] = gate_positive*in_positive
    bins['gate+_in-'] = gate_positive*~in_positive
    bins['gate-_in+'] = (~gate_positive)*in_positive
    bins['gate-_in-'] = (~gate_positive)*~in_positive
    for case,zero_one in bins.items():
        intermediate[(case, 'freq')] = _get_reduce(zero_one, 'sum')
        for key_to_summarise in VALUES_TO_SUMMARISE:
            for reduction in REDUCTIONS:
                if key_to_summarise.startswith('hook'):
                    values = cache[f'mlp.{key_to_summarise}']
                # elif key_to_summarise.startswith('gate'):
                #     values = cache['mlp.hook_post']*bins[key_to_summarise]
                elif key_to_summarise=='swish':
                    values = model.actfn(cache['mlp.hook_pre'])
                else:
                    raise NotImplementedError(key_to_summarise)
                values *= zero_one
                intermediate[(case, key_to_summarise, reduction)] = _get_reduce(
                    values,
                    reduction=reduction,
                    arg=(reduction!="sum"),
                )
                #batch pos layer neuron -> {'values': batch layer neuron, 'indices': batch layer neuron}
    return intermediate

def get_all_neuron_acts_on_dataset(
    args, model, dataset, path=None
):
    """Get all neuron activations on dataset.

    Args:
        args (Namespace): The argparse arguments
        model (HookedTransformer): The model to run
        dataset (Dataset): A Huggingface-style dataset to run the model on
        path (str, optional): The path to save the data.
            Within this path we will have a subdirectory activation_cache.
            Defaults to None (i.e., current directory).

    Returns:
        dict[Tensor]: a dict of tensors with all the relevant information
            (cached activations and summary statistics).
        Keys are those in the KEYS constant.
    """
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
    if path is None:
        path = '.'

    batched_dataset = dataset.batch(
        batch_size=args.batch_size,
        drop_last_batch=False
        ) #preserves order

    names_filter = [
        f"blocks.{layer}.{hook}"
        for layer in range(model.cfg.n_layers)
        for hook in HOOKS_TO_CACHE
    ]

    if not os.path.exists(f'{path}/activation_cache'):
        os.mkdir(f'{path}/activation_cache')
    for i, batch in tqdm(enumerate(batched_dataset)):
        if i<args.resume_from or os.path.exists(f"{path}/activation_cache/batch{i}.pickle"):
            with open(f"{path}/activation_cache/batch{i}.pickle", 'rb') as f:
                intermediate = _move_to(pickle.load(f), device='cuda')
        else:
            intermediate = _get_all_neuron_acts(model, batch, names_filter, dataset.max_seq_len)
            with open(f"{path}/activation_cache/batch{i}.pickle", 'wb') as f:
                pickle.dump(_move_to(intermediate, 'cpu'), f)
        del intermediate['ln_cache']
        if i==0:
            out_dict={}
            for key,value in intermediate.items():
                if key[-1] in ['sum', 'freq']:
                    out_dict[key]=value
                elif key[-1] in ['max', 'min']:
                    out_dict[key] = {
                        'values':value['values'],
                        'indices':torch.full_like(value['values'], 0)
                    }
        else:
            for key,value in out_dict.items():
                if key[-1] in ['sum', 'freq']:
                    out_dict[key] = _get_reduce(
                        torch.stack([value, intermediate[key]]),
                        'sum'
                        )#batch layer neuron -> layer neuron
                elif key[-1] in ['max', 'min']:
                    out_dict[key] = {
                        'values': torch.cat(
                            [
                                value['values'],
                                intermediate[key]['values']
                                ]
                            ),
                        'indices':torch.cat(
                            [
                                value['indices'],
                                torch.full_like(intermediate[key]['values'], i)
                                ]
                            )
                    }#both entries: sample layer neuron
                    if i>=args.examples_per_neuron:#running topk computation
                        #print(out_dict[key]['values'].shape) #should be: k layer neuron
                        vi = _get_reduce(
                            out_dict[key]['values'],
                            reduction=key[1],
                            arg=True,
                            k=args.examples_per_neuron,
                            )#k+1 layer neuron -> k layer neuron
                        out_dict[key]['values'] = vi['values']
                        out_dict[key]['indices'] = torch.gather(
                            out_dict[key]['indices'], dim=0, index=vi['indices']
                        )
                        #original dataset indices!
                        #I want:
                        #new_out_dict[key]['indices'][i,layer,neuron] =
                        # out_dict[key]['indices'][vi['indices'][i,layer,neuron],layer,neuron]
                        #hence the above line of code

    for key in out_dict:
        if key[-1]=='sum':
            #for the moment frequencies are still absolute numbers so we can do this
            out_dict[key] = out_dict[key].to(torch.float) / out_dict[(key[0],'freq')].to(torch.float)
            #now the 'sum' entry is actually a mean!
    for key in out_dict:
        if key[-1]=='freq':
            out_dict[key] = out_dict[key].to(torch.float) / float(dataset.n_tokens)

    return out_dict

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='dolma-small')
    parser.add_argument('--model', default='allenai/OLMo-1B-hf')
    parser.add_argument(
        '--refactor_glu',
        action='store_true',
        help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
    )
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--examples_per_neuron', default=16, type=int)
    parser.add_argument('--resume_from', default=0)
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--save_to', default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.save_to:
        RUN_CODE = args.save_to
    elif args.test:
        RUN_CODE = "test"
    else:
        RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset}"
    #OLMO-1B-hf_dolma-v1_7-3B

    SAVE_PATH = f"{args.results_dir}/{RUN_CODE}"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    torch.set_grad_enabled(False)

    model = ModelWrapper.from_pretrained(args.model, refactor_glu=args.refactor_glu)

    dataset = datasets.load_from_disk(f'{args.datasets_dir}/{args.dataset}')
    assert isinstance(dataset, datasets.Dataset)
    if args.test:
        dataset = dataset.select(range(2))
    add_properties(dataset)

    print('computing activations...')
    SUMMARY_FILE = f'{SAVE_PATH}/summary{"_refactored" if args.refactor_glu else""}.pickle'
    if not os.path.exists(f'{SAVE_PATH}/summary.pickle'):
        out_dict = get_all_neuron_acts_on_dataset(
            args=args,
            model=model,
            dataset=dataset,
            path=SAVE_PATH,
        )
        with open(f'{SAVE_PATH}/summary.pickle', 'wb') as f:
            pickle.dump(_move_to(out_dict, 'cpu'), f)
    print('done!')
