#!/bin/bash
#SBATCH --mem=10gb
#SBATCH -c2
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --time=6:00:00
#SBATCH --array=0-24%24
#SBATCH --killable
#SBATCH --requeue

python spectral_detuning.py --subset="vit" --output_path="./recovered_weights/vit/" \
--start_layer=$SLURM_ARRAY_TASK_ID --n_layers_to_recover=1 --sched_end_rank=16 --n_loras=5