from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
import einops
from transformer_lens import HookedTransformer
from datasets import load_from_disk

from b_activations import _move_to
from b2_recompute import recompute_acts
from c_neuron_vis import neuron_vis_full
import utils

parser = ArgumentParser()
parser.add_argument('--dataset', default='dolma-small')
parser.add_argument('--model', default='allenai/OLMo-1B-hf')
parser.add_argument(
    '--refactor_glu',
    action='store_true',
    help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
)
parser.add_argument('--datasets_dir', default='datasets')
parser.add_argument('--results_dir', default='results')
parser.add_argument('--save_to', default=None)
parser.add_argument('--neurons',
    nargs='+',
    default=[],
    help='one or several neurons denoted as layer.neuron, or "all"',
)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

print("preparation...")
if args.save_to:
    RUN_CODE = args.save_to
elif args.test:
    RUN_CODE = "test"
else:
    RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset}"
#OLMO-1B-hf_dolma-v1_7-3B
#the id of the b_activations.py run

SAVE_PATH = f"{args.results_dir}/{RUN_CODE}"
TITLE = f"<h1>Model: <b>{args.model}</b></h1>\n"

torch.set_grad_enabled(False)

SUMMARY_FILE = f'{SAVE_PATH}/summary{"_refactored" if args.refactor_glu else""}.pickle'
if refactored_already:= os.path.exists(SUMMARY_FILE):
    MY_FILE = SUMMARY_FILE
else:
    MY_FILE = f'{SAVE_PATH}/summary.pickle'
with open(MY_FILE, 'rb') as f:
    summary_dict = _move_to(pickle.load(f), 'cuda')
    print(f"summary_dict: {summary_dict.keys()}")

dataset = load_from_disk(f'{args.datasets_dir}/{args.dataset}')

model = HookedTransformer.from_pretrained(
    args.model,
    refactor_glu=args.refactor_glu and refactored_already,#not yet refactor_glu=args.refactor_glu
    device='cpu' if (args.refactor_glu and not refactored_already) else 'cuda',
)
tokenizer = model.tokenizer
if args.refactor_glu and not refactored_already:
    #first we detect which neurons to refactor and then we update the model
    sign_to_adapt = torch.sign(einops.einsum(
        model.W_in.detach().cuda(), model.W_gate.detach().cuda(), "l d n, l d n -> l n"
    ))
    summary_dict = utils.refactor_glu(summary_dict, sign_to_adapt)
    with open(SUMMARY_FILE, 'wb') as f:
        pickle.dump(summary_dict, f)
    del model
    model = HookedTransformer.from_pretrained(args.model, refactor_glu=True, device='cuda')

summary_dict['max_activations']['indices'] = summary_dict['max_activations']['indices'].to(torch.int)
summary_dict['min_activations']['indices'] = summary_dict['min_activations']['indices'].to(torch.int)

maxmin_indices = torch.cat(
    [summary_dict['max_activations']['indices'], summary_dict['min_activations']['indices']]
)#.to(torch.int)

TOPK, N_LAYERS, N_NEURONS = summary_dict['max_activations']['indices'].shape
if args.neurons=='all':
    layer_neuron_list = [range(N_NEURONS) for _layer in range(N_LAYERS)]
elif args.neurons:
    layer_neuron_list = [[] for layer in range(N_LAYERS)]
    for ln_str in args.neurons:
        layer, neuron = tuple(int(n) for n in ln_str.split('.'))
        layer_neuron_list[layer].append(neuron)
elif args.test:
    layer_neuron_list = [[0]]

for layer,neuron_list in enumerate(layer_neuron_list):
    print(f'processing layer {layer}...')
    layer_dir = f"{SAVE_PATH}/L{layer}"
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    for neuron in tqdm(neuron_list):
        neuron_dir = f"{layer_dir}/N{neuron}"
        if not os.path.exists(neuron_dir):
            os.mkdir(neuron_dir)
        #recomputing neuron activations on max and min examples
        activations_file = f'{neuron_dir}/activations{"_refactored" if args.refactor_glu else ""}.pt'
        activations_file_raw = f'{neuron_dir}/activations.pt'
        if os.path.exists(activations_file):
            dict_all = torch.load(activations_file)
        elif args.refactor_glu and os.path.exists(activations_file_raw):
            dict_all = torch.load(activations_file_raw)
            if sign_to_adapt[layer,neuron]==-1:
                dict_all = utils.adapt_activations(dict_all)
            torch.save(dict_all, activations_file)
        else:
            dict_all = recompute_acts(
                model,
                layer, neuron,
                maxmin_indices[:,layer,neuron],
                # out_dict=dict_all,#dict_all is just updated
                save_path=SAVE_PATH,
            )
            torch.save(dict_all, activations_file)
        #visualisation
        neuron_data = {
            'max_indices':maxmin_indices[:TOPK, layer, neuron],
            'min_indices':maxmin_indices[TOPK:, layer, neuron],
            'max_acts':dict_all['acts'][:TOPK],
            'min_acts':dict_all['acts'][TOPK:],
            'max_val':summary_dict['max_activations']['values'][0,layer,neuron],
            'min_val':summary_dict['min_activations']['values'][0,layer,neuron],
            'avg_val':summary_dict['summary_mean'][layer,neuron],
            'act_freq':summary_dict['summary_freq'][layer,neuron],
            'argmax_tokens':summary_dict['max_activations']['indices'][:,layer,neuron],
            'argmin_tokens':summary_dict['min_activations']['indices'][:,layer,neuron],
        }
        # We add some text to tell us what layer and neuron we're looking at
        heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
        HTML = TITLE + heading + neuron_vis_full(
                neuron_data=neuron_data,
                dataset=dataset,
                tokenizer=tokenizer,
        )
        with open(f'{neuron_dir}/vis.html', 'w', encoding="utf-8") as f:
            f.write(HTML)
print('done!')
