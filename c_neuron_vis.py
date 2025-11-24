"""Functions to visualise a neuron once all the data is computed"""

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

def neuron_vis_full(activation_data, dataset, model):
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
    # We add a kind of
    # "table": cases are main columns,
    # and within that we have paragraphs with frequency,
    # and max/min/mean (all within one paragraph)
    # of gate/swish/in/post (separate paragraphs)
    htmls.append('<table><tr>')
    htmls.extend([f"<td><h4>{case}</h4></td>" for case in CASES])
    htmls.append('</tr><tr>')
    htmls.extend([f"<td>Frequency: <b>{activation_data[(case, 'freq')]:.2%}</b>.</td>" for case in CASES])
    htmls.append('</tr>')
    extreme_values = {}
    maxima = {}
    minima = {}
    for act_type in VALUES_TO_SUMMARISE:
        for case in CASES:
            if (case,act_type,'max') in activation_data.keys():
                extreme_values[(case, act_type)] = activation_data[(case,act_type,'max')]['values'][0]
            elif act_type=='swish':
                extreme_values[(case, act_type)] = model.actfn(extreme_values[(case, 'hook_pre')])
            maxima[(case, act_type)] = extreme_values[(case, act_type)] if extreme_values[(case, act_type)]>0 else 0
            minima[(case, act_type)] = extreme_values[(case, act_type)] if extreme_values[(case, act_type)]<0 else -0
        htmls.append('<tr>')
        htmls.extend(
            [f"""<td>
            <b>{act_type}</b>:<br>
            Max: <b>{maxima[(case, act_type)]:.2f}</b>;<br>
            Min: <b>{minima[(case, act_type)]:.2f}</b>;<br>
            Avg: <b>{activation_data[(case,act_type,'sum')]:.2f}</b>.
            </td>
            """
            for case in CASES
            ]
        )
        htmls.append('</tr>')
    htmls.append('</table>')
    #TODO possibility to toggle the lists
    for case in CASES:
        htmls.append(f'<h2>Prototypical activations for case {case}</h2>')
        for act_type in VALUES_TO_SUMMARISE:
            key = (case, act_type, 'max')
            if key in activation_data and 'all_acts' in activation_data[key] and activation_data[key]['values'][0]!=0:
                htmls.append(f'<h3>Extreme {act_type} activations')
                for i in range(activation_data[key]['indices'].shape[0]):
                    # print(max_indices[i])
                    # print(dataset[int(max_indices[i])])
                    #ignore samples in which no token satisfies the condition:
                    first_acts=activation_data[key]['all_acts'][i,:,0]#sample pos act_type
                    # if case=='gate+_in-' and act_type=='hook_post':
                    #     print(first_acts)
                    if allclose(first_acts, zeros_like(first_acts), atol=1e-7):
                        break
                    htmls.append(
                        _vis_example(
                            i=i,
                            indices=activation_data[key]['indices'],
                            acts=activation_data[key]['all_acts'],
                            stop_tokens=activation_data[key]['position_indices']+3,
                            dataset=dataset,
                            tokenizer=model.tokenizer,
                            key=key,
                            )
                    )
            htmls.append('<hr>')
        htmls.append('<hr>')
    return "\n".join(htmls)
