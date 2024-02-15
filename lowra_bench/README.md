
# LoWRA Bench
This benchmark consists of:
- [`inference`](./inference/) - scripts for running inference on the recovered Pre-FT weights.
- [`eval`](./eval/) - scripts for running the evaluation metrics on the generated outputs.    


## Using the Recovered Pre-Fine-Tuning Weights
To run inference on the Pre-FT recovered weights use the following scripts:
### ViT: 
1. Download the ImageNet validation subset from [https://drive.google.com/file/d/1l1HT3lZ31wkxtCLuX5CB2eTLTKAu-v9j/view?usp=sharing](https://drive.google.com/file/d/1l1HT3lZ31wkxtCLuX5CB2eTLTKAu-v9j/view?usp=sharing).
2. Extract the dataset into `datasets/imagenet_val_5k`
3. Run the inference script:
```bash
python inference/vit_inference.py --input_path="../recovered_weights/vit/"
```

### Stable Diffusion: 
```bash
python inference/stable_diffusion_inference.py --input_path="../recovered_weights/stable_diffusion_15/"
```

### Mistral SFT: 
```bash
python inference/mistral_inference.py --input_path="../recovered_weights/mistral7b_01_sft/" --subset="mistral-7b-v0.1-sft"
```

### Mistral DPO: 
```bash
python inference/mistral_inference.py --input_path="../recovered_weights/mistral7b_01_dpo/" --subset="mistral-7b-v0.1-dpo"
```

## Evaluating the Recovered Weights 
We evaluate the success of a Pre-FT weight recovery method using semantic metrics. 
The evaluation is performed by comparing the generated outputs of the original Pre-FT model
to the results of the recovered one under the same seed. 

To generate the results using the Pre-FT model add the `--gen_pre_ft_model` argument to the scripts above.

To generate the results using the fine-tuned LoRA models add the `--gen_finetuned_models` argument to the above scripts.

Once all the results are generated, you can run the semantic metrics located under the [`eval`](./eval/) dir:
### LPIPS Evaluation:
```bash 
python eval/run_lpips.py --pre_ft_images_path="../recovered_weights/stable_diffusion_15/generated_images/pre_ft" \
        --target_images_path="../recovered_weights/stable_diffusion_15/generated_images/recovered_model" \
        --output_path="../recovered_weights/stable_diffusion_15/lpips_results"
```

### SBERT Evaluation:
```bash 
# Mistral SFT
python eval/run_sbert_similarity.py --pre_ft_text_path="../recovered_weights/mistral7b_01_sft/generated_text/generated_pre_ft.json" \
        --target_text_path="../recovered_weights/mistral7b_01_sft/generated_text/generated_recovered.json" \
        --output_path="../recovered_weights/mistral7b_01_sft/sbert_results"

# Mistral DPO        
python eval/run_sbert_similarity.py --pre_ft_text_path="../recovered_weights/mistral7b_01_dpo/generated_text/generated_pre_ft.json" \
        --target_text_path="../recovered_weights/mistral7b_01_dpo/generated_text/generated_recovered.json" \
        --output_path="../recovered_weights/mistral7b_01_dpo/sbert_results"
```
