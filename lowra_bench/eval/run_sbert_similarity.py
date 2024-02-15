import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

import sys

sys.path.append('../')
from utils import fix_seeds


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_ft_text_path", type=str, required=True)
    parser.add_argument("--target_text_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = define_arguments()
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sbert_model.eval()
    sbert_model.requires_grad_(False)

    fix_seeds(args)
    with torch.no_grad():
        with open(args.pre_ft_text_path, 'r') as file:
            pre_ft_generated_text = json.load(file)
        pre_ft_generated_text = list(pre_ft_generated_text.values())
        pre_ft_embeddings = sbert_model.encode(pre_ft_generated_text, convert_to_tensor=True)

        with open(args.target_text_path, 'r') as file:
            target_generated_text = json.load(file)
        target_generated_text = list(target_generated_text.values())
        target_embeddings = sbert_model.encode(target_generated_text, convert_to_tensor=True)

        assert pre_ft_embeddings.shape == target_embeddings.shape

        sbert_dists = 1.0 - F.cosine_similarity(pre_ft_embeddings, target_embeddings).clip(-1.0, 1.0)

        results = []
        for prompt_idx in tqdm(range(len(sbert_dists))):
            results.append({
                "sbert_dist": sbert_dists[prompt_idx].item(),
                "sbert_dist_log10": np.log10(sbert_dists[prompt_idx].item() + np.finfo(np.float64).eps),
                "prompt_idx": prompt_idx,
            })

        results_df = pd.DataFrame(results)
        os.makedirs(args.output_path, exist_ok=True)
        results_df.to_csv(os.path.join(args.output_path, f"sbert.csv"), index=False)
        print(f"Log10 SBERT Mean={results_df['sbert_dist_log10'].mean()}, Log10 SBERT STD={results_df['sbert_dist_log10'].std()}")
        print(f"SBERT Mean={results_df['sbert_dist'].mean()}, SBERT STD={results_df['sbert_dist'].std()}")
