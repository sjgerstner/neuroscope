#TODO adapt to changes in activations code
# (pickle vs pt, keys)

from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
import einops
from transformer_lens import HookedTransformer
from datasets import load_from_disk

from utils import _move_to
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

model = utils.ModelWrapper.from_pretrained(
    args.model,
    refactor_glu=args.refactor_glu and refactored_already,#not yet refactor_glu=args.refactor_glu
    device='cpu' if (args.refactor_glu and not refactored_already) else 'cuda',
)
assert model.W_gate is not None

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
    model = utils.ModelWrapper.from_pretrained(args.model, refactor_glu=True, device='cuda')
#TOPK = summary_dict[('gate+_in+', 'max')]['indices'].shape[0]#topk layer neuron
if args.neurons=='all':
    layer_neuron_list = [range(model.cfg.d_mlp) for _layer in range(model.cfg.n_layers)]
elif args.neurons:
    layer_neuron_list = [[] for layer in range(model.cfg.n_layers)]
    for ln_str in args.neurons:
        layer, neuron = tuple(int(n) for n in ln_str.split('.'))
        layer_neuron_list[layer].append(neuron)
elif args.test:
    layer_neuron_list = [[0]]

maxmin_keys = [key for key in summary_dict.keys() if key[-1] in ['max','min']]
maxmin_indices = torch.cat(
    [summary_dict[key]['indices'] for key in maxmin_keys]
)

TOPK, N_LAYERS, N_NEURONS = summary_dict[('gate+_in+', 'hook_post', 'max')]['indices'].shape
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
        #TODO we may need to comment this out because the internal format changed
            neuron_data = torch.load(activations_file)
        elif args.refactor_glu and os.path.exists(activations_file_raw):
            neuron_data = torch.load(activations_file_raw)
            if sign_to_adapt[layer,neuron]==-1:
                neuron_data = utils.adapt_activations(neuron_data)
            torch.save(neuron_data, activations_file)
        else:
            dict_all = recompute_acts(
                model,
                layer, neuron,
                maxmin_indices[:,layer,neuron],
                # out_dict=dict_all,#dict_all is just updated
                save_path=SAVE_PATH,
            )
            neuron_data = {
                recomputed_type_key : {
                    case_key : stacked_tensor[i*TOPK:(i+1)*TOPK]
                    for i,case_key in enumerate(maxmin_keys)
                }
                for recomputed_type_key,stacked_tensor in dict_all.items()
            }
            #outer key is the recomputed activation type, inner key is (case, summarised_type, ['max','min'])
            #the other way round doesn't work because of what comes next!
            torch.save(neuron_data, activations_file)
        #visualisation
        for key,value in summary_dict.items():
            if isinstance(value, torch.Tensor):
                neuron_data[key]=value[...,layer,neuron]
            elif isinstance(value, dict):
                neuron_data[key]={key1:value1[...,layer,neuron] for key1,value1 in value.items()}
        # neuron_data = {
        #     'max_indices':maxmin_indices[:TOPK, layer, neuron],
        #     'min_indices':maxmin_indices[TOPK:, layer, neuron],
        #     'max_acts':dict_all['acts'][:TOPK],
        #     'min_acts':dict_all['acts'][TOPK:],
        #     'max_val':summary_dict[('hook_post', 'max')]['values'][0,layer,neuron],
        #     'min_val':summary_dict[('hook_post', 'min')]['values'][0,layer,neuron],
        #     'avg_val':summary_dict['mean'][layer,neuron],
        #     'act_freq':summary_dict['summary_freq'][layer,neuron],
        #     'argmax_tokens':summary_dict[('hook_post', 'max')]['indices'][:,layer,neuron],
        #     'argmin_tokens':summary_dict[('hook_post', 'min')]['indices'][:,layer,neuron],
        # }
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
