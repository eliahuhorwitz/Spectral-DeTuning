import gc
import re
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import set_seed
from transformers import pipeline
from datasets import load_dataset

import sys

sys.path.append('../')
from utils import rsetattr, fix_seeds, merge_lora_weights


def generate_pre_ft(args, prompts):
    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        pipe = pipeline("text-generation", model=args.pre_ft_model, torch_dtype=torch.bfloat16, device_map="auto")

        generated_outputs = {}
        for prompt_idx, prompt in enumerate(prompts):
            set_seed(args.seed)
            outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_outputs[prompt_idx] = outputs[0]["generated_text"]

        with open(os.path.join(args.output_path, f"generated_pre_ft.json"), 'w') as generated_file:
            json.dump(generated_outputs, generated_file, indent=4)


def generate_finetuned_models(args, prompts):
    os.makedirs(args.output_path, exist_ok=True)
    dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
    dataset = dataset.with_format("torch")["train"]
    total_n_loras = 15  # Note: In the LoWRA Bench dataset, each subset has 15 different loras
    lora_names = [dataset[0][f'lora_{lora_idx}_name'] for lora_idx in range(total_n_loras)]
    assert len(dataset) == 128
    for lora_idx, lora_name in enumerate(lora_names):
        fix_seeds(args)
        with torch.no_grad():
            lora_pipe = pipeline("text-generation", model=args.pre_ft_model, torch_dtype=torch.bfloat16, device_map="auto")
            for layer_idx in tqdm(range(len(dataset)), desc=f'Loading lora {lora_idx} weights...'):
                layer = merge_lora_weights(dataset, layer_idx, lora_idx)
                rsetattr(lora_pipe.model, layer['layer_name'], torch.nn.Parameter(layer[f'lora_{lora_idx}_merged_weights'].to("cuda").to(torch.bfloat16), requires_grad=False))

            generated_outputs = {}
            for prompt_idx, prompt in enumerate(prompts):
                set_seed(args.seed)
                outputs = lora_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                generated_outputs[prompt_idx] = outputs[0]["generated_text"]

            with open(os.path.join(args.output_path, f"generated_{lora_name}.json"), 'w') as generated_file:
                json.dump(generated_outputs, generated_file, indent=4)


def generate_recovered_model(args, prompts):
    fix_seeds(args)
    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        lora_pipe = pipeline("text-generation", model=args.pre_ft_model, torch_dtype=torch.bfloat16, device_map="auto")

        layer_name_pattern = re.compile(r'layer_(\d{4})')
        all_dirs = [folder for folder in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, folder))]
        layer_dirs = [folder for folder in all_dirs if layer_name_pattern.match(folder)]
        assert len(layer_dirs) == 128
        for layer_dir in tqdm(layer_dirs, desc=f'Merging weights...'):
            recovered_layer = torch.load(os.path.join(args.input_path, layer_dir, f"W_star.pt"))
            rsetattr(lora_pipe.model, recovered_layer['layer_name'], torch.nn.Parameter(recovered_layer[f'recovered_weight'].to("cuda").to(torch.bfloat16), requires_grad=False))

        generated_outputs = {}
        for prompt_idx, prompt in enumerate(prompts):
            set_seed(args.seed)
            outputs = lora_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_outputs[prompt_idx] = outputs[0]["generated_text"]

        with open(os.path.join(args.output_path, f"generated_recovered.json"), 'w') as generated_file:
            json.dump(generated_outputs, generated_file, indent=4)

        lora_pipe = None
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="path to the recovered weights")
    parser.add_argument("--prompts_path", type=str, default="./alpaca_farm_eval.csv", help="path to a dataframe with a list of prompts to evaluate on")
    parser.add_argument("--pre_ft_model", type=str, default="mistralai/Mistral-7B-v0.1", help="path or hugging face model name of the pre-fine-tuning model")

    parser.add_argument("--subset", type=str, choices=["mistral-7b-v0.1-sft", "mistral-7b-v0.1-dpo"], required=True,
                        help="LoWRA Bench dataset subset, options are 'mistral-7b-v0.1-sft' or 'mistral-7b-v0.1-dpo'")
    parser.add_argument("--dataset", type=str, default="Eliahu/LoWRA-Bench", help="dataset path, supports hugging face datasets")
    parser.add_argument("--cache_dir", type=str, default="../../.cache/lowra_bench", help="path to cache the dataset, prevents downloading the entire dataset for every layer in distributed mode")

    parser.add_argument('--gen_pre_ft_model', action='store_true', help="generate samples for the original, pre-fine-tuning model")
    parser.add_argument('--gen_finetuned_models', action='store_true', help="generate samples for all the 15 finetuned lora models")
    parser.add_argument("--seed", type=int, default=0, help="fix the seed for generation")

    return parser.parse_args()


if __name__ == '__main__':
    args = define_arguments()
    fix_seeds(args)
    prompts = pd.read_csv(args.prompts_path)["0"].tolist()
    device = torch.device('cuda')

    args.output_path = os.path.join(args.input_path, "generated_text")
    os.makedirs(args.output_path, exist_ok=True)

    args_file_path = os.path.join(args.output_path, f"args.json")
    with open(args_file_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    generate_recovered_model(args=args, prompts=prompts)

    if args.gen_pre_ft_model:
        generate_pre_ft(args=args, prompts=prompts)
    if args.gen_finetuned_models:
        generate_finetuned_models(args=args, prompts=prompts)
