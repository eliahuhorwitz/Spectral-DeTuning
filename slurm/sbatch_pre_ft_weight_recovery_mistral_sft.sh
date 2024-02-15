#!/bin/bash
#SBATCH --mem=10gb
#SBATCH -c2
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --time=6:00:00
#SBATCH --array=0-128%128
#SBATCH --killable
#SBATCH --requeue

python spectral_detuning.py --subset="mistral-7b-v0.1-sft" --output_path="./recovered_weights/mistral7b_01_sft/" \
--start_layer=$SLURM_ARRAY_TASK_ID --n_layers_to_recover=1 --sched_end_rank=64 --n_loras=12 --n_iters=1000