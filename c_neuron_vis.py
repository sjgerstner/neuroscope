"""Functions to visualise a neuron once all the data is computed"""

# from argparse import ArgumentParser
# import os
# import pickle
# from tqdm import tqdm

# from datasets import load_from_disk
# from transformers import AutoTokenizer

from torch import allclose, zeros_like
from circuitsvis.tokens import colored_tokens_multi

from utils import CASES, get_act_type_keys
from b_activations import VALUES_TO_SUMMARISE

def _vis_example(i, indices, acts, dataset, tokenizer, key, stop_tokens=None):
    index = int(indices[i])
    #print(dataset[index]['input_ids'])#tensor of ints
    tokens = tokenizer.batch_decode(#or convert_ids_to_tokens
        dataset[index]['input_ids']
    )
    if stop_tokens is not None:
        #TODO truncate beginning
        #TODO option to show full example
        tokens = tokens[:stop_tokens[i]]
    relevant_acts = acts[i,:len(tokens),:]#batch, pos, act_type
    return f'<h4>Example {i}</h4>\n'+str(#TODO source of dataset example
        colored_tokens_multi(
            tokens=tokens,
            values=relevant_acts,
            labels=get_act_type_keys(key),
        )
    )+"\n</div>"

def neuron_vis_full(neuron_data, dataset, tokenizer):
    """Full neuron visualisation for a given neuron.
    Args:
        neuron_data (dict): contains summary statistics and data on max/min activations
        dataset (datasets.Dataset)
        tokenizer (Huggingface tokenizer)
    Returns:
        html string
    """
    htmls = []
    # # We first add the style to make each token element have a nice border
    # htmls = [style_string]
    #TODO weight-based analysis (circuitsvis topk tokens + RW functionality)
    #TODO improve table layout
    # We add a kind of
    # "table": cases are main columns,
    # and within that we have paragraphs with frequency,
    # and max/min/mean (all within one paragraph)
    # of gate/swish/in/post (separate paragraphs)
    htmls.append('<table><tr>')
    for case in CASES:
        htmls.append(f"<td><h4>{case}</h4>")
        htmls.append(f"<p>Frequency: <b>{neuron_data[(case, 'freq')]:.2%}</b>.</p>")
        for act_type in VALUES_TO_SUMMARISE:
            htmls.append(
                f"""<p>
                <b>{act_type}</b>:
                Max: <b>{neuron_data[(case,act_type,'max')]['values'][0]:.4f}</b>;
                Min: <b>{neuron_data[(case,act_type,'min')]['values'][0]:.4f}</b>;
                Avg: <b>{neuron_data[(case,act_type,'sum')]:.4f}</b>.
                </p>
                """#TODO fewer digits
            )
        htmls.append('</td>')
    htmls.append('</tr></table>')
    #TODO possibility to toggle the lists
    for case in CASES:
        htmls.append(f'<h2>Prototypical activations for case {case}</h2>')
        for act_type in VALUES_TO_SUMMARISE:
            for reduction in ['max','min']:
                key = (case, act_type, reduction)
                if neuron_data[key]['values'][0]!=0:
                    htmls.append(f'<h3>{reduction} {act_type} activations')
                    for i in range(neuron_data[key]['indices'].shape[0]):
                        # print(max_indices[i])
                        # print(dataset[int(max_indices[i])])
                        #ignore samples in which no token satisfies the condition:
                        first_acts=neuron_data[key]['all_acts'][i,:,0]#sample pos act_type
                        # if case=='gate+_in-' and act_type=='hook_post':
                        #     print(first_acts)
                        if allclose(first_acts, zeros_like(first_acts), atol=1e-7):
                            break
                        htmls.append(
                            _vis_example(
                                i=i,
                                indices=neuron_data[key]['indices'],
                                acts=neuron_data[key]['all_acts'],
                                stop_tokens=neuron_data[key]['position_indices']+3,
                                dataset=dataset,
                                tokenizer=tokenizer,
                                key=key,
                                )
                        )
            htmls.append('<hr>')
        htmls.append('<hr>')
    return "\n".join(htmls)

# if __name__=="__main__":
#     parser = ArgumentParser()
#     parser.add_argument('--datadir', default=None)
#     parser.add_argument('--model', default='allenai/OLMo-1B-hf')
#     parser.add_argument('--dataset',
#                         default='dolma_small',
#                         help='pretokenized dataset'
#                         )
#     parser.add_argument('--save_to',
#                         default=None,
#                         help='should be the same as the corresponding b_activations.py run.'
#                         )
#     parser.add_argument('--test', action='store_true')
#     args = parser.parse_args()

#     if args.save_to:
#         RUN_CODE = args.save_to
#     elif args.test:
#         RUN_CODE = "test"
#     else:
#         RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset.split('-')[0]}"
#     #OLMO-1B-hf_dolma-v1_7-3B
#     #the id of the b-activations.py run

#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     dataset = load_from_disk(f'{args.datadir}/{args.dataset}')

#     with open(f'{args.datadir}/{RUN_CODE}/summary.pickle', 'rb') as f:
#         summary_dict = pickle.load(f)
#         print(f"summary_dict: {summary_dict.keys()}")
#     with open(f'{args.datadir}/{RUN_CODE}/indices.pickle', 'rb') as f:
#         maxmin_indices = pickle.load(f) #sample layer neuron
#         print(f'maxmin_indices: {maxmin_indices.shape}')
#     with open(f'{args.datadir}/{RUN_CODE}/activations.pickle', 'rb') as f:
#         activations_dict = pickle.load(f)
#         print(f"activations_dict['acts']: {activations_dict['acts'].shape}")

#     TOPK = activations_dict['acts'].shape[2]//2
#     if args.test:
#         NLAYERS=1
#         NNEURONS=1
#     else:
#         NLAYERS=activations_dict['acts'].shape[0]
#         NNEURONS=activations_dict['acts'].shape[1]

#     TITLE = f"<h1>Model: <b>{args.model}</b></h1>\n"

#     for layer in range(NLAYERS):
#         if not os.path.exists(f"{args.datadir}/{RUN_CODE}/L{layer}"):
#             os.mkdir(f"{args.datadir}/{RUN_CODE}/L{layer}")
#         for neuron in tqdm(range(NNEURONS)):
#             neuron_data = {
#                 'max_indices':maxmin_indices[:TOPK, layer, neuron],
#                 'min_indices':maxmin_indices[TOPK:, layer, neuron],
#                 'max_acts':activations_dict['acts'][layer,neuron, :TOPK],
#                 'min_acts':activations_dict['acts'][layer,neuron, TOPK:],
#                 'max_val':summary_dict['summary_max'][layer,neuron],
#                 'min_val':summary_dict['summary_min'][layer,neuron],
#                 'avg_val':summary_dict['summary_mean'][layer,neuron],
#                 'act_freq':summary_dict['summary_freq'][layer,neuron],
#                 'argmax_tokens':summary_dict['argmax_activations'][:,layer,neuron],
#                 'argmin_tokens':summary_dict['argmin_activations'][:,layer,neuron],
#             }
#             # We add some text to tell us what layer and neuron we're looking at
#             heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
#             HTML = TITLE + heading + neuron_vis_full(
#                    neuron_data=neuron_data,
#                    dataset=dataset,
#                    tokenizer=tokenizer,
#             )
#             with open(
#                 f'{args.datadir}/{RUN_CODE}/L{layer}/N{neuron}.html',
#                 'w',
#                 encoding="utf-8",
#             ) as f:
#                 f.write(HTML)
#     print('done!')
