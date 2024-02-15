import re
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, AutoImageProcessor
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor, )

import sys

sys.path.append('../')
from utils import rsetattr, fix_seeds, merge_lora_weights



def load_imagenet_dataset(args):
    image_processor = AutoImageProcessor.from_pretrained(args.pre_ft_model)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    val_transforms = Compose([Resize(image_processor.size["height"]), CenterCrop(image_processor.size["height"]), ToTensor(), normalize, ])
    imagenet_dataset = ImageFolder(args.eval_dataset_path, transform=val_transforms)
    return imagenet_dataset


def eval(pre_ft_model, target_model, eval_dataset, batch_size=64, max_layer=12):
    with torch.no_grad():
        dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        distances = [0] * max_layer

        for images, _ in dataloader:
            images = images.cuda()
            actual_batch_size = images.shape[0]

            pre_ft_outputs = pre_ft_model(images, output_attentions=True, output_hidden_states=True).hidden_states
            recovered_outputs = target_model(images, output_attentions=True, output_hidden_states=True).hidden_states

            for inter_layer_idx in range(max_layer):
                temp_pre_ft_outputs = pre_ft_outputs[inter_layer_idx].reshape(actual_batch_size, -1)
                temp_recovered_outputs = recovered_outputs[inter_layer_idx].reshape(actual_batch_size, -1)
                distances[inter_layer_idx] += (1 - F.cosine_similarity(temp_pre_ft_outputs, temp_recovered_outputs)).mean().item()

        distances = [distance / len(dataloader) for distance in distances]
        return distances


def eval_finetuned_models(args, imagenet_dataset):
    with torch.no_grad():
        pre_ft_model = ViTForImageClassification.from_pretrained(args.pre_ft_model, ignore_mismatched_sizes=True).to("cuda")
        pre_ft_model.eval()
        dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
        dataset = dataset.with_format("torch")["train"]
        total_n_loras = 15  # Note: In the LoWRA Bench dataset, each subset has 15 different loras
        lora_names = [dataset[0][f'lora_{lora_idx}_name'] for lora_idx in range(total_n_loras)]
        assert len(dataset) == 24
        for lora_idx, lora_name in enumerate(lora_names):
            representation_results = []
            lora_model = ViTForImageClassification.from_pretrained(args.pre_ft_model, ignore_mismatched_sizes=True).to("cuda")
            lora_model.eval()
            for layer_idx in tqdm(range(len(dataset)), desc=f'Loading lora {lora_idx} weights...'):
                layer = merge_lora_weights(dataset, layer_idx, lora_idx)
                rsetattr(lora_model, layer['layer_name'], torch.nn.Parameter(layer[f'lora_{lora_idx}_merged_weights'].to("cuda"), requires_grad=False))

            rep_dists = eval(pre_ft_model=pre_ft_model, target_model=lora_model, eval_dataset=imagenet_dataset)
            for layer_idx, rep_dist in enumerate(rep_dists):
                representation_results.append({"rep_dist": rep_dist, "layer_idx": layer_idx, "output_path_root": args.output_path_root, "experiment_name": f"lora_init_{lora_name}", })
            pd.DataFrame(representation_results).to_csv(os.path.join(args.output_path_root, f"rep_dist_lora_init_{lora_name}.csv"), index=False)


def eval_recovered_model(args, imagenet_dataset):
    with torch.no_grad():
        pre_ft_model = ViTForImageClassification.from_pretrained(args.pre_ft_model, ignore_mismatched_sizes=True).to("cuda")
        pre_ft_model.eval()

        representation_results = []
        recovered_model = ViTForImageClassification.from_pretrained(args.pre_ft_model, ignore_mismatched_sizes=True).to("cuda")
        recovered_model.eval()

        layer_name_pattern = re.compile(r'layer_(\d{4})')
        all_dirs = [folder for folder in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, folder))]
        layer_dirs = [folder for folder in all_dirs if layer_name_pattern.match(folder)]
        assert len(layer_dirs) == 24

        for layer_dir in tqdm(layer_dirs, desc=f'Merging weights...'):
            recovered_layer = torch.load(os.path.join(args.input_path, layer_dir, f"W_star.pt"))
            rsetattr(recovered_model, recovered_layer['layer_name'], torch.nn.Parameter(recovered_layer[f'recovered_weight'].to("cuda"), requires_grad=False))

        rep_dists = eval(pre_ft_model=pre_ft_model, target_model=recovered_model, eval_dataset=imagenet_dataset)
        for layer_idx, rep_dist in enumerate(rep_dists):
            representation_results.append({
                "rep_dist": rep_dist,
                "layer_idx": layer_idx,
                "output_path_root": args.output_path_root,
                "experiment_name": f"recovered_loras",
            })
        pd.DataFrame(representation_results).to_csv(os.path.join(args.output_path_root, f"rep_dist_recovered.csv"), index=False)


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="path to the recovered weights")
    parser.add_argument("--eval_dataset_path", type=str, default="../datasets/imagenet_val_5k", help="root path for the eval dataset")
    parser.add_argument("--pre_ft_model", type=str, default="google/vit-base-patch16-224", help="path or hugging face model name of the pre-fine-tuning model")

    parser.add_argument("--subset", type=str, choices=["vit"], help="LoWRA Bench dataset subset, options are 'vit'")
    parser.add_argument("--dataset", type=str, default="Eliahu/LoWRA-Bench", help="dataset path, supports hugging face datasets")
    parser.add_argument("--cache_dir", type=str, default="../../.cache/lowra_bench", help="path to cache the dataset, prevents downloading the entire dataset for every layer in distributed mode")

    parser.add_argument('--gen_finetuned_models', action='store_true', help="generate samples for all the 15 finetuned lora models")
    parser.add_argument("--seed", type=int, default=0, help="fix the seed for generation")

    return parser.parse_args()


if __name__ == '__main__':
    args = define_arguments()
    fix_seeds(args)
    device = torch.device('cuda')

    args.output_path_root = os.path.join(args.input_path, "semantic_metrics")
    os.makedirs(args.output_path_root, exist_ok=True)
    imagenet_val_dataset = load_imagenet_dataset(args)

    args_file_path = os.path.join(args.output_path_root, f"args.json")
    with open(args_file_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    eval_recovered_model(args=args, imagenet_dataset=imagenet_val_dataset)
    if args.gen_finetuned_models == 1:
        eval_finetuned_models(args=args, imagenet_dataset=imagenet_val_dataset)


