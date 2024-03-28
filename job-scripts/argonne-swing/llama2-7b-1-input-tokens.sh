#!/bin/bash

#SBATCH -J llama2-7b-1-input-tokens

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(128 256 512 1024 2048 4096)
HF_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="llama2-7b"
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
module load amd-uprof
cd /home/ac.gwilkins/energy-inference/cuda/

# for num_tokens in "${N_TOKENS[@]}"
# 	do 
mkdir ./$MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f ./$MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid
    # done
