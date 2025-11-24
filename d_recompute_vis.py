from argparse import ArgumentParser
import os
import pickle

import torch
import einops
from datasets import load_from_disk

import recompute
from c_neuron_vis import neuron_vis_full
import utils
from utils import _move_to

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
parser.add_argument('--site_dir', default='website')
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
VIS_PATH = f"{args.site_dir}/{RUN_CODE}"
TITLE = f"<h1>Model: <b>{args.model}</b></h1>\n"

torch.set_grad_enabled(False)

SUMMARY_FILE = f'{SAVE_PATH}/summary{"_refactored" if args.refactor_glu else""}'
if refactored_already:= os.path.exists(f"{SUMMARY_FILE}.pt") or os.path.exists(f"{SUMMARY_FILE}.pickle"):
    MY_FILE = SUMMARY_FILE
else:
    MY_FILE = f'{SAVE_PATH}/summary'
if os.path.exists(f"{MY_FILE}.pt"):
    summary_dict = torch.load(f"{MY_FILE}.pt")
else:
    assert os.path.exists(f"{MY_FILE}.pickle")
    with open(f"{MY_FILE}.pickle", 'rb') as f:
        summary_dict = _move_to(pickle.load(f), 'cuda')
# if args.test:
#     #print(f"summary_dict: {summary_dict.keys()}")
#     for key,value in summary_dict.items():
#         if isinstance(value, torch.Tensor):
#             print(f'{key}: {value[...,0,0]}')
#         elif isinstance(value, dict):
#             print(f'{key}:')
#             for key1,value1 in value.items():
#                 print(f'> {key1}: {value1[...,0,0]}')

dataset = load_from_disk(f'{args.datasets_dir}/{args.dataset}')

model = utils.ModelWrapper.from_pretrained(
    args.model,
    refactor_glu=args.refactor_glu and refactored_already,#not yet refactor_glu=args.refactor_glu
    device='cpu' if (args.refactor_glu and not refactored_already) else 'cuda',
)
assert model.W_gate is not None

tokenizer = model.tokenizer
TOPK, N_LAYERS, N_NEURONS = summary_dict[('gate+_in+', 'hook_post', 'max')]['indices'].shape
if args.refactor_glu and not refactored_already:
    #first we detect which neurons to refactor and then we update the model
    sign_to_adapt = torch.sign(einops.einsum(
        model.W_in.detach().cuda(), model.W_gate.detach().cuda(), "l d n, l d n -> l n"
    ))
    summary_dict = utils.refactor_glu(summary_dict, sign_to_adapt)
    torch.save(summary_dict, f"{SUMMARY_FILE}.pt")
    del model
    model = utils.ModelWrapper.from_pretrained(args.model, refactor_glu=True, device='cuda')
else:
    sign_to_adapt = torch.ones(size=(N_LAYERS, N_NEURONS), dtype=torch.int)
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
# maxmin_indices = torch.cat(
#     [summary_dict[key]['indices'] for key in maxmin_keys]
# )

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
    for neuron in neuron_list:
        print(f'> processing neuron {neuron}...')
        neuron_dir = f"{layer_dir}/N{neuron}"
        if not os.path.exists(neuron_dir):
            os.mkdir(neuron_dir)
        #recomputing neuron activations on max and min examples
        print('>> gathering/recomputing data from cache...')
        activation_data = recompute.recompute_acts_if_necessary(
            args=args,
            summary_dict=summary_dict,
            maxmin_keys=maxmin_keys,
            neuron_dir=neuron_dir,
            single_sign_to_adapt=int(sign_to_adapt[layer,neuron]),
            model=model, layer=layer, neuron=neuron,
            save_path=SAVE_PATH,
        )
        activation_data = recompute.expand_with_summary(
            activation_data=activation_data,
            summary_dict=summary_dict,
            layer=layer, neuron=neuron,
        )
        #visualisation
        print('>> creating html page...')
        # We add some text to tell us what layer and neuron we're looking at
        heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
        HTML = TITLE + heading + neuron_vis_full(
                activation_data=activation_data,
                dataset=dataset,
                model=model,
        )
        with open(f'{VIS_PATH}/L{layer}/N{neuron}/vis.html', 'w', encoding="utf-8") as f:
            f.write(HTML)
print('done!')
