import re
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline


import sys
sys.path.append('../')
from utils import rgetattr, rsetattr, fix_seeds, merge_lora_weights


def generate_pre_ft(args, prompts, device, batch_size=5):
    fix_seeds(args)
    images_save_path_root = os.path.join(args.output_path, "pre_ft")
    os.makedirs(images_save_path_root, exist_ok=True)
    with torch.no_grad():
        pipe = StableDiffusionPipeline.from_pretrained(args.pre_ft_model, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        all_latents = pipe.prepare_latents(len(prompts), 4, 512, 512, pipe.dtype, device, generator=generator)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        for prompt_idx in range(0, len(prompts), batch_size):
            images = pipe(latents=all_latents[prompt_idx:prompt_idx + batch_size], prompt=prompts[prompt_idx:prompt_idx + batch_size], num_images_per_prompt=1, generator=generator)[0]
            for image_idx, image in enumerate(images):
                image.save(os.path.join(images_save_path_root, f"{prompt_idx + image_idx}.png"))


def generate_finetuned_models(args, prompts, device, batch_size=5):
    dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
    dataset = dataset.with_format("torch")["train"]
    total_n_loras = 15  # Note: In the LoWRA Bench dataset, each subset has 15 different loras
    lora_names = [dataset[0][f'lora_{lora_idx}_name'] for lora_idx in range(total_n_loras)]
    assert len(dataset) == 264
    for lora_idx, lora_name in enumerate(lora_names):
        fix_seeds(args)
        images_save_path_root = os.path.join(args.output_path, f"lora_{lora_name.split('/')[-1].split('.')[0]}")
        os.makedirs(images_save_path_root, exist_ok=True)
        with torch.no_grad():
            lora_pipe = StableDiffusionPipeline.from_pretrained(args.pre_ft_model, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
            for layer_idx in tqdm(range(len(dataset)), desc=f'Loading lora {lora_idx} weights...'):
                layer = merge_lora_weights(dataset, layer_idx, lora_idx)
                original_layer = rgetattr(lora_pipe, f"{layer['layer_model']}.{layer['layer_name']}")
                rsetattr(lora_pipe, f"{layer['layer_model']}.{layer['layer_name']}",
                         torch.nn.Parameter(layer[f'lora_{lora_idx}_merged_weights'].reshape(original_layer.shape).to("cuda"), requires_grad=False))

            generator = torch.Generator(device=device).manual_seed(args.seed)
            all_latents = lora_pipe.prepare_latents(len(prompts), 4, 512, 512, lora_pipe.dtype, device, generator=generator)

            generator = torch.Generator(device=device).manual_seed(args.seed)

            for prompt_idx in range(0, len(prompts), batch_size):
                images = lora_pipe(latents=all_latents[prompt_idx:prompt_idx + batch_size], prompt=prompts[prompt_idx:prompt_idx + batch_size], num_images_per_prompt=1, generator=generator)[0]
                for image_idx, image in enumerate(images):
                    image.save(os.path.join(images_save_path_root, f"{prompt_idx + image_idx}.png"))


def generate_recovered_model(args, prompts, device, batch_size=5):
    fix_seeds(args)
    save_path_root = os.path.join(args.output_path, "recovered_model")
    os.makedirs(save_path_root, exist_ok=True)
    with torch.no_grad():
        lora_pipe = StableDiffusionPipeline.from_pretrained(args.pre_ft_model, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
        layer_name_pattern = re.compile(r'layer_(\d{4})')
        all_dirs = [folder for folder in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, folder))]
        layer_dirs = [folder for folder in all_dirs if layer_name_pattern.match(folder)]
        assert len(layer_dirs) == 264
        for layer_dir in tqdm(layer_dirs, desc=f'Merging weights...'):
            recovered_layer = torch.load(os.path.join(args.input_path, layer_dir, f"W_star.pt"))
            original_layer = rgetattr(lora_pipe, f"{recovered_layer['layer_model']}.{recovered_layer['layer_name']}")
            rsetattr(lora_pipe, f"{recovered_layer['layer_model']}.{recovered_layer['layer_name']}",
                     torch.nn.Parameter(recovered_layer[f'recovered_weight'].reshape(original_layer.shape).to("cuda"), requires_grad=False))

        generator = torch.Generator(device=device).manual_seed(args.seed)
        all_latents = lora_pipe.prepare_latents(len(prompts), 4, 512, 512, lora_pipe.dtype, device, generator=generator)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        for prompt_idx in range(0, len(prompts), batch_size):
            images = lora_pipe(latents=all_latents[prompt_idx:prompt_idx + batch_size],
                               prompt=prompts[prompt_idx:prompt_idx + batch_size], num_images_per_prompt=1, generator=generator)[0]
            for image_idx, image in enumerate(images):
                image.save(os.path.join(save_path_root, f"{prompt_idx + image_idx}.png"))


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="path to the recovered weights")
    parser.add_argument("--prompts_path", type=str, default="./coco_prompts.csv", help="path to a dataframe with a list of prompts to evaluate on")
    parser.add_argument("--pre_ft_model", type=str, default="runwayml/stable-diffusion-v1-5", help="path or hugging face model name of the pre-fine-tuning model")

    parser.add_argument("--subset", type=str, default="stable-diffusion-1.5", help="LoWRA Bench dataset subset")
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

    args.output_path = os.path.join(args.input_path, "generated_images")
    os.makedirs(args.output_path, exist_ok=True)

    args_file_path = os.path.join(args.output_path, f"args.json")
    with open(args_file_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    generate_recovered_model(args=args, prompts=prompts, device=device)

    if args.gen_pre_ft_model:
        generate_pre_ft(args=args, prompts=prompts, device=device)

    if args.gen_finetuned_models:
        generate_finetuned_models(args=args, prompts=prompts, device=device)
