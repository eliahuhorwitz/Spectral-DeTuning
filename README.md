# Recovering the Pre-Fine-Tuning Weights of Generative Models
Official PyTorch Implementation for the "Recovering the Pre-Fine-Tuning Weights of Generative Models" paper.  
<p align="center">
    🌐 <a href="https://vision.huji.ac.il/spectral_detuning/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2402.10208" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/datasets/Eliahu/LoWRA-Bench" target="_blank">Dataset</a> <br>
</p>

![](imgs/header.gif)


***Pre-Fine-Tuning Weight Recovery Attack Setting:***  We uncover a vulnerability in LoRA fine-tuned models wherein an attacker is 
able to undo the fine-tuning process and recover the weights of the original pre-trained model. 
The setting for the vulnerability is as follows: 

(a) The attacker only has access to n different LoRA fine-tuned models. 

(b) The attacker assumes that all n models originated from the same source model.  

(c) Using only the n visible models, the attacker attempts to recover the original source model.

Our method, *Spectral DeTuning*, can perform the attack in an unsupervised and data-free manner on real models such as Stable Diffusion and Mistral. 
For simplicity, we illustrate the attack on a single layer, in reality, the attack is carried out independently on all the fine-tuned layers.

**Note: The attacker has no access to the low-rank decomposition of the fine-tuned models.**
___

> **Recovering the Pre-Fine-Tuning Weights of Generative Models**<br>
> Eliahu Horwitz, Jonathan Kahana, Yedid Hoshen<br>
> <a href="https://arxiv.org/abs/2402.10208" target="_blank">https://arxiv.org/abs/2402.10208</a> <br>
>
>**Abstract:** The dominant paradigm in generative modeling consists of two steps: 
> i) pre-training on a large-scale but unsafe dataset, ii) aligning the pre-trained model with human values via fine-tuning.
> This practice is considered safe, as no current method can recover the unsafe, *pre-fine-tuning* model weights. 
> In this paper, we demonstrate that this assumption is often false. Concretely, we present *Spectral DeTuning*, 
> a method that can recover the weights of the pre-fine-tuning model using a few low-rank (LoRA) fine-tuned models. 
> In contrast to previous attacks that attempt to recover pre-fine-tuning capabilities, 
> our method aims to recover the exact pre-fine-tuning weights. 
> Our approach exploits this new vulnerability against large-scale models such as a personalized Stable Diffusion and an aligned Mistral.


## Project Structure
This project consists of:
- `spectral_detuning.py` - main file for recovering the Pre-FT weights using Spectral DeTuning.
- `distributed_spectral_detuning.py` - Distributing Spectral DeTuning across multiple CPU cores of a single machine.
- `increase_rank_on_plateau_scheduler.py` - rank scheduler class.
- [`slurm`](./slurm/) - Examples for distributing Spectral DeTuning across a slurm cluster.   
- [`lowra_bench`](./lowra_bench/) - Scripts for running inference and evaluation of the recovered weights.   


## Installation 
1.  Clone the repo:
```bash
git clone https://github.com/eliahuhorwitz/spectral_detuning.git
cd spectral_detuning
```
2. Create a new environment and install the libraries:
```bash
python3 -m venv spectral_detuning_venv
source spectral_detuning_venv/bin/activate
pip install -r requirements.txt
```



## Running Spectral DeTuning for Pre-Fine-Tuning Weight Recovery 
The `spectral_detuning.py` script is the main script in this project. 
It handles the downloading of the LoWRA Bench dataset that is hosted 
on Hugging Face.

Below are examples for running runs Spectral DeTuning for Pre-FT weight recovery on the 
LoWRA Bench dataset subset using different distribution strategies.   

### Single GPU Execution
These use a single GPU to recover all the layers one by one *sequentially*.

#### ViT
```bash
python spectral_detuning.py --subset="vit" --output_path="./recovered_weights/vit/" \
--start_layer=0 --n_layers_to_recover=-1 --sched_end_rank=16 --n_loras=5 
```
> [!TIP] 
> ViT contains 24 layers to recover and can be recovered *sequentially* in a few minutes on a desktop grade GPU.

#### Stable Diffusion
```bash
python spectral_detuning.py --subset="stable-diffusion-1.5" \ 
--output_path="./recovered_weights/stable_diffusion_15/" --start_layer=0 \
--n_layers_to_recover=-1 --sched_end_rank=32 --n_loras=5 
```
> [!IMPORTANT] 
> Stable Diffusion contains 264 layers to recover. See below for a faster option.

#### Mistral SFT
```bash
python spectral_detuning.py --subset="mistral-7b-v0.1-sft" \
--output_path="./recovered_weights/mistral7b_01_sft/" --start_layer=0 \
--n_layers_to_recover=-1 --sched_end_rank=64 --n_loras=12 --n_iters=1000 
```

#### Mistral DPO
```bash
python spectral_detuning.py --subset="mistral-7b-v0.1-dpo" \
--output_path="./recovered_weights/mistral7b_01_dpo/" --start_layer=0 \
--n_layers_to_recover=-1 --sched_end_rank=64 --n_loras=8 --n_iters=1000
```
> [!IMPORTANT] 
> Mistral contains 128 layers to recover, some of them are of high dimensions (up to 4096x4096), see below for a faster option.


### Distributed Multiprocess CPU Execution
Since Spectral DeTuning does not require gradients or running 
inference on the model, it can run quickly even on a CPU. 
Below are options for distributing Spectral DeTuning across the CPU cores
of a single machine using multiple processes.     

To run using this strategy, run `distributed_spectral_detuning.py` with the same arguments as above.
To control the number of CPU cores to distribute across use the `--n_cpus` argument,
set `--n_cpus=-1` to use all available core.
> [!TIP] 
> ViT contains 24 layers to recover and can be recovered in minutes when distributed across desktop CPU cores.


### Distributed Execution on a Compute Cluster
In cases where the model has many layers (e.g., Stable Diffusion and Mistral),
it is recommended to distribute the recovery across a compute cluster (GPU or CPU).
We provide example slurm scripts under the [`slurm`](./slurm/) dir. 

The main difference is the `--n_layers_to_recover` argument which controls how many layers
each machine will recover.

> [!TIP] 
> Spectral DeTuning can recover a *single layer* of a large model (e.g. Mistral-7B)
> in under 5 minutes on a *single desktop GPU* (e.g. RTX2080). 
> The recovery speed of the entire model is a function of the number of machines in your cluster.   



## Using the Recovered Pre-Fine-Tuning Weights
To run inference on the Pre-FT recovered weights use the following scripts:
#### ViT: 
```bash
python lowra_bench/inference/vit_inference.py --input_path="./recovered_weights/vit/"
```

#### Stable Diffusion: 
```bash
python lowra_bench/inference/stable_diffusion_inference.py \
--input_path="./recovered_weights/stable_diffusion/"
```

#### Mistral SFT: 
```bash
python lowra_bench/inference/mistral_inference.py \
--input_path="./recovered_weights/mistral7b_01_sft/" --subset="mistral-7b-v0.1-sft"
```

#### Mistral DPO: 
```bash
python lowra_bench/inference/mistral_inference.py \
--input_path="./recovered_weights/mistral7b_01_dpo/" --subset="mistral-7b-v0.1-dpo"
```


## Using a Custom Dataset of Fine-tuned LoRAs and Pre-FT Models
Coming soon...
- [ ] Preprocessing scripts for constructing a LoRA dataset similar to the LoWRA Bench one.



## Citation
If you find this useful for your research, please use the following.

```
@article{horwitz2024recovering,
  title={Recovering the Pre-Fine-Tuning Weights of Generative Models},
  author={Horwitz, Eliahu and Kahana, Jonathan and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2402.10208},
  year={2024}
}
```


## Acknowledgments
- The project makes extensive use of the different Hugging Face libraries (e.g. [Diffusers](https://huggingface.co/docs/diffusers/en/index), [PEFT](https://huggingface.co/docs/peft/en/index), [Transformers](https://huggingface.co/docs/transformers/en/index)).
- The [LoWRA Bench dataset](https://huggingface.co/datasets/Eliahu/LoWRA-Bench) is hosted on Hugging Face.
- The fine-tuning of Mistral was performed based on the Zephyr model as seen [here](https://github.com/huggingface/alignment-handbook/tree/main).
- The fine-tuned LoRA models for Stable Diffusion are taken from civitai and were fine-tuned by [RalFinger](https://civitai.com/user/RalFinger).
- The rank scheduler is based on the PyTorch [ReduceLROnPlateau Scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
