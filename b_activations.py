"""
Compute max activations for all neurons on a given dataset.
The code was written with batch size 1 in mind.
Other batch sizes will almost certainly lead to bugs.
"""

#TODO enable other batch sizes
#TODO (also other files) pathlib


from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
#from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import einops
#from transformers import DataCollatorWithPadding
import datasets

from utils import (
    ModelWrapper,
    _move_to,
    add_properties,
    VALUES_TO_SUMMARISE,
    RELEVANT_SIGNS,
    detect_cases
)

HOOKS_TO_CACHE = ['ln2.hook_normalized', 'mlp.hook_post', 'mlp.hook_pre', 'mlp.hook_pre_linear']
REDUCTIONS = ['max', 'sum']

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
            vi_dict[key] = einops.rearrange(
                tensor, '... 1 layer neuron -> ... layer neuron'
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

    batch_size = len(ids_and_mask['input_ids'])
    seq_len = max(len(ids) for ids in ids_and_mask['input_ids'])

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
    bins=detect_cases(gate_values=cache['mlp.hook_pre'], in_values=cache['mlp.hook_pre_linear'])
    for case,zero_one in bins.items():
        intermediate[(case, 'freq')] = _get_reduce(zero_one, 'sum')
        for key_to_summarise in VALUES_TO_SUMMARISE:
            if key_to_summarise.startswith('hook'):
                values = cache[f'mlp.{key_to_summarise}']
            elif key_to_summarise=='swish' and case.startswith('gate-'):
                values = model.actfn(cache['mlp.hook_pre'])
            else:
                continue
            relevant_values = values*zero_one
            for reduction in REDUCTIONS:
                intermediate[(case, key_to_summarise, reduction)] = _get_reduce(
                    relevant_values if reduction=="sum" else torch.abs(relevant_values),
                    reduction=reduction,
                    arg=(reduction!="sum"),
                )
                #batch pos layer neuron -> {'values': batch layer neuron, 'indices': batch layer neuron}
                if reduction=='max':
                    intermediate[(case, key_to_summarise, reduction)]['values'] *= RELEVANT_SIGNS[case][key_to_summarise]
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

    # collator = DataCollatorWithPadding(
    #     tokenizer=None,          # we don’t need a tokenizer because the fields are already ints
    #     padding=True,            # pad to the longest sequence in the batch
    #     max_length=None,         # you can set a hard max_length if you wish
    #     pad_to_multiple_of=None, # optional, useful for certain hardware constraints
    # )
    batched_dataset = dataset.batch(
        batch_size=args.batch_size,
        drop_last_batch=False
        ) #preserves order
    # batched_dataset = DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    #     collate_fn=collator,
    # )#preserves order

    names_filter = [
        f"blocks.{layer}.{hook}"
        for layer in range(model.cfg.n_layers)
        for hook in HOOKS_TO_CACHE
    ]

    if not os.path.exists(f'{path}/activation_cache'):
        os.mkdir(f'{path}/activation_cache')
    previous_batch_size = 0
    if os.path.exists(f'{path}/activation_cache/batch_size.txt'):
        with open(f'{path}/activation_cache/batch_size.txt', 'r', encoding='utf-8') as f:
            previous_batch_size = int(f.read())
    #print(previous_batch_size, args.batch_size)
    batch_size_unchanged = previous_batch_size==args.batch_size
    if not batch_size_unchanged:
        with open(f'{path}/activation_cache/batch_size.txt', 'w', encoding='utf-8') as f:
            f.write(str(args.batch_size))
    for i, batch in tqdm(enumerate(batched_dataset)):
        # if i==0:
        #     print(batch)
        batch = {
            'input_ids': pad_sequence(batch['input_ids'], padding_value=model.tokenizer.pad_token_type_id),
            'attention_mask': pad_sequence(batch['attention_mask'])
        }
        batch_file = f"{path}/activation_cache/batch{i}"
        if batch_size_unchanged and os.path.exists(f"{batch_file}.pt"):
            intermediate = torch.load(f"{batch_file}.pt")
        elif batch_size_unchanged and os.path.exists(f"{batch_file}.pickle"):
            with open(f"{batch_file}.pickle", 'rb') as f:
                intermediate = _move_to(pickle.load(f), device='cuda')
        else:
            intermediate = _get_all_neuron_acts(model, batch, names_filter, dataset.max_seq_len)
            torch.save(intermediate, f"{batch_file}.pt")
        del intermediate['ln_cache']
        #TODO the following stuff could probably be more efficient, even the base case i==0
        if i==0:
            out_dict={}
            for key,value in intermediate.items():
                if key[-1] in ['sum', 'freq']:
                    out_dict[key]=value
                elif key[-1] in ['max', 'min']:
                    out_dict[key] = {
                        'values':value['values'],
                        'indices':torch.stack([
                            torch.full((model.cfg.n_layers,model.cfg.n_neurons), counter)
                            for counter in range(args.batch_size)
                        ])
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
                                torch.stack([
                                        torch.full((model.cfg.n_layers,model.cfg.n_neurons), i*args.batch_size+counter)
                                        for counter in range(args.batch_size)
                                    ])
                                ]
                            )
                    }#both entries: sample layer neuron
                    #running topk computation
                    #print(out_dict[key]['values'].shape) #should be: k layer neuron
                    vi = _get_reduce(
                        out_dict[key]['values'],
                        reduction=key[-1],
                        arg=True,
                        k=min(i+1, args.examples_per_neuron),
                        )#k+1 layer neuron -> k layer neuron
                    # if args.test:
                    #     print(out_dict[key]['indices'].shape)
                    #     print(vi['indices'].shape)
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
        if key[-1] in ('sum', 'freq'):
            out_dict[key] = out_dict[key].to(torch.float)
    for key in out_dict:
        if key[-1]=='sum':
            #for the moment frequencies are still absolute numbers so we can do this
            out_dict[key] /= out_dict[(key[0],'freq')]
            #now the 'sum' entry is actually a mean!
    for key in out_dict:
        if key[-1]=='freq':
            out_dict[key] /= float(dataset.n_tokens)

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
    #parser.add_argument('--resume_from', default=0)
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
        dataset = dataset.select(range(4))
    add_properties(dataset)
    # dataset = dataset.with_format(
    #     type="torch",
    #     columns=["input_ids", "attention_mask"],
    #     pad=True,                # <-- enable automatic padding
    #     padding_value=model.tokenizer.pad_token_type_id,         # match your model's pad token
    #     pad_to_multiple_of=None
    # )

    print('computing activations...')
    SUMMARY_FILE = f'{SAVE_PATH}/summary{"_refactored" if args.refactor_glu else""}'
    if not os.path.exists(f'{SUMMARY_FILE}.pickle') and not os.path.exists(f'{SUMMARY_FILE}.pt'):
        out_dict = get_all_neuron_acts_on_dataset(
            args=args,
            model=model,
            dataset=dataset,
            path=SAVE_PATH,
        )
        torch.save(out_dict, f'{SUMMARY_FILE}.pt')
    print('done!')
