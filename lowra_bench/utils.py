import os
import torch
import random
import functools
import numpy as np
from copy import deepcopy
from pytorch_lightning import seed_everything

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def fix_seeds(args):
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.


def merge_lora_weights( dataset, layer_idx, lora_idx):
    layer = deepcopy(dataset[layer_idx])
    merged_layer = {}

    merged_layer['layer_model'] = layer['layer_model']
    merged_layer['layer_name'] = layer['layer_name']
    merged_layer['pre_ft_name'] = layer['pre_ft_name']
    W_pre_ft = deepcopy(layer['pre_ft_weight'])
    merged_layer['pre_ft_weight'] = deepcopy(W_pre_ft)

    alpha = layer[f'lora_{lora_idx}_alpha']
    rank = layer[f'lora_{lora_idx}_rank']
    B = deepcopy(layer[f'lora_{lora_idx}_B_weight'])
    A = deepcopy(layer[f'lora_{lora_idx}_A_weight'])

    merged_layer[f'lora_{lora_idx}_name'] = layer[f'lora_{lora_idx}_name']
    merged_layer[f'lora_{lora_idx}_rank'] = layer[f'lora_{lora_idx}_rank']
    merged_layer[f'lora_{lora_idx}_alpha'] = layer[f'lora_{lora_idx}_alpha']
    merged_layer[f'lora_{lora_idx}_merged_weights'] = W_pre_ft + ((alpha / rank * B) @ A)

    assert not torch.allclose(merged_layer[f'lora_{lora_idx}_merged_weights'], W_pre_ft)
    return merged_layer