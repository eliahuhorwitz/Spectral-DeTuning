import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
from pytorch_lightning import seed_everything
from increase_rank_on_plateau_scheduler import IncreaseRankOnPlateau

def fix_seeds(args):
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.


def merge_lora_weights(args, layer_idx, device):
    # Note: Load the huggingface dataset and extract a single layer with multiple LoRA models
    dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
    layer = deepcopy(dataset.with_format("torch")["train"][layer_idx])

    merged_layer = {}

    # Note: load the ground truth weights
    merged_layer['layer_model'] = layer['layer_model']
    merged_layer['layer_name'] = layer['layer_name']
    merged_layer['pre_ft_name'] = layer['pre_ft_name']
    W_pre_ft = deepcopy(layer['pre_ft_weight']).to(device).float()
    merged_layer['pre_ft_weight'] = deepcopy(W_pre_ft)

    # Note: merge the LoRA weights for all existing LoRA models
    for lora_idx in args.lora_ids:
        alpha = layer[f'lora_{lora_idx}_alpha']
        rank = layer[f'lora_{lora_idx}_rank']
        B = deepcopy(layer[f'lora_{lora_idx}_B_weight']).to(device).float()
        A = deepcopy(layer[f'lora_{lora_idx}_A_weight']).to(device).float()

        merged_layer[f'lora_{lora_idx}_name'] = layer[f'lora_{lora_idx}_name']
        merged_layer[f'lora_{lora_idx}_rank'] = rank
        merged_layer[f'lora_{lora_idx}_alpha'] = alpha
        merged_layer[f'lora_{lora_idx}_merged_weights'] = W_pre_ft + ((alpha / rank * B) @ A)

        assert torch.allclose(merged_layer['pre_ft_weight'], layer['pre_ft_weight'].to(device).float())
        assert not torch.allclose(merged_layer[f'lora_{lora_idx}_merged_weights'], layer['pre_ft_weight'].to(device).float())
        assert not torch.allclose(merged_layer[f'lora_{lora_idx}_merged_weights'], merged_layer['pre_ft_weight'])
    return merged_layer


def calc_loss(W_primes, W_star, M_s):
    losses = [torch.mean((W_primes[lora_idx] - (W_star + M_s[lora_idx])) ** 2) for lora_idx in range(len(W_primes))]
    loss = torch.mean(torch.stack(losses, axis=0), axis=0).item()
    return loss


def recover_layer(args, layer_idx, device):
    output_root = os.path.join(args.output_path, f"layer_{layer_idx:04}")
    args.output_root = output_root
    os.makedirs(args.output_root, exist_ok=True)
    layer = merge_lora_weights(args, layer_idx, device)

    W_pre_ft = layer["pre_ft_weight"]
    W_primes = [layer[f'lora_{lora_idx}_merged_weights'] for lora_idx in args.lora_ids]

    curr_rank = args.sched_start_rank
    rank_sched = IncreaseRankOnPlateau(n_iters=args.n_iters, start_rank=curr_rank, end_rank=args.sched_end_rank)

    with torch.no_grad():
        W_star = torch.mean(torch.stack(W_primes, axis=0), axis=0)

        for iter in tqdm(range(args.n_iters), desc=f"Recovering weights for layer {layer_idx}"):
            M_s = [W_prime - W_star for W_prime in W_primes]
            for i in range(len(M_s)):
                (U, S, V) = torch.svd_lowrank(M_s[i], q=curr_rank)
                M_s[i] = (U @ torch.diag_embed(S)) @ V.T
            W_star = torch.mean(torch.stack([W_prime - M_i for (W_prime, M_i) in zip(W_primes, M_s)], axis=0), axis=0)

            # Calculate and print losses
            gt_error = torch.mean((W_star - W_pre_ft) ** 2).item()
            loss = calc_loss(W_primes, W_star, M_s)
            print(f"Iter {iter} | Loss: {loss} | GT Error: {gt_error} | Rank: {curr_rank}")

            rank_sched.step(loss)
            curr_rank = rank_sched.curr_rank

        torch.save({"layer_name": layer["layer_name"], "layer_model": layer['layer_model'], "recovered_weight": W_star},
                   os.path.join(args.output_root, f"W_star.pt"))


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, choices=["vit", "stable-diffusion-1.5", "mistral-7b-v0.1-sft", "mistral-7b-v0.1-dpo"], required=True,
                        help="LoWRA Bench dataset subset, options are 'vit', 'stable-diffusion-1.5', 'mistral-7b-v0.1-sft', 'mistral-7b-v0.1-dpo'")
    parser.add_argument("--dataset", type=str, default="Eliahu/LoWRA-Bench", help="dataset path, supports hugging face datasets")
    parser.add_argument("--cache_dir", type=str, default="./.cache/lowra_bench", help="path to cache the dataset, prevents downloading the entire dataset for every layer in distributed mode")

    parser.add_argument("--output_path", type=str, required=True, help="path to save the recovered weights")
    parser.add_argument("--start_layer", type=int, required=True, help="layer id to start the recovery from")
    parser.add_argument("--n_layers_to_recover", type=int, default=1, help="the number of layers to recover, -1 to recover the weight of all the layers in the dataset subset")
    parser.add_argument("--n_loras", type=int, default=5, help="number of fine-tuned loras to use for recovering the weights")
    parser.add_argument("--lora_ids", nargs="*", type=int, default=[],
                        help="the lora ids to use (as they appear in the LoWRA Bench dataset), must be the same size as n_loras or empty to sample a random subset of size n_loras")

    parser.add_argument("--n_iters", type=int, default=300, help="number of optimization steps per layer")
    parser.add_argument("--sched_start_rank", type=int, default=1, help="the starting rank for the rank scheduler")
    parser.add_argument("--sched_end_rank", type=int, default=16, help="the end rank for the rank scheduler")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


if __name__ == '__main__':
    args = define_args().parse_args()
    fix_seeds(args)
    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)

    total_n_loras = 15  # Note: In the LoWRA Bench dataset, each subset has 15 different loras
    if len(args.lora_ids) == 0:
        args.lora_ids = random.sample(range(total_n_loras), args.n_loras)

    # Note: Load the huggingface dataset
    dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
    dataset = dataset.with_format("torch")["train"]

    if args.n_layers_to_recover == -1: # Note: Recover all the layers in the dataset
        end_layer = len(dataset)
    else:
        end_layer = min(args.start_layer + args.n_layers_to_recover, len(dataset))
    for layer_idx in range(args.start_layer, end_layer):
        recover_layer(args, layer_idx=layer_idx, device=device)
