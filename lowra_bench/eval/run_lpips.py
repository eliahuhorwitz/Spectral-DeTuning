import os
import lpips
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import sys

sys.path.append('../')
from utils import fix_seeds


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_ft_images_path", type=str, required=True)
    parser.add_argument("--target_images_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_images_from_path(image_dir_path):
    images_pil = []
    for image_name in os.listdir(image_dir_path):
        if not image_name.endswith(".png"):
            continue
        image_path = os.path.join(image_dir_path, image_name)
        image = Image.open(image_path)
        images_pil.append(image)

    images_pil.sort(key=lambda x: x.filename)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  #
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    images_torch = torch.stack([preprocess(image) for image in images_pil]).cuda()
    return images_torch


if __name__ == '__main__':
    args = define_arguments()

    lpips_loss = lpips.LPIPS(net='vgg').cuda()
    lpips_loss.eval()
    lpips_loss.requires_grad_(False)

    fix_seeds(args)
    with torch.no_grad():
        pre_ft_images = load_images_from_path(args.pre_ft_images_path)
        target_images = load_images_from_path(args.target_images_path)
        assert len(pre_ft_images) == len(target_images)
        results = []
        for image_idx in tqdm(range(len(pre_ft_images))):
            pre_ft_image = pre_ft_images[image_idx].unsqueeze(0)
            target_image = target_images[image_idx].unsqueeze(0)
            loss = lpips_loss(pre_ft_image, target_image)
            results.append({
                "lpips_loss": loss.item(),
                "image_idx": image_idx,
                "pre_ft_images_path": args.pre_ft_images_path,
                "target_images_path": args.target_images_path,
            })

        results_df = pd.DataFrame(results)
        os.makedirs(args.output_path, exist_ok=True)
        results_df.to_csv(os.path.join(args.output_path, f"lpips.csv"), index=False)
        print(f"LPIPS Mean={results_df['lpips_loss'].mean()}, LPIPS STD={results_df['lpips_loss'].std()}")
