#!/bin/bash

#SBATCH -J mistral-7b-2-input-tokens

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:2

N_NODES=1
N_GPUS_PER_NODE=2
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(128 256 512 1024 2048 4096)
HF_NAME="mistralai/Mistral-7B-v0.1"
MODEL_NAME="mistral-7b"
#BATCH_SIZES=(8 16 64 128)
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
module load amd-uprof

cd /home/ac.gwilkins/energy-inference/cuda/

for num_tokens in "${N_TOKENS[@]}"
	do 
        AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 open-source-cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens $num_tokens --hf_name $HF_NAME --system_name $SYSTEM
    done