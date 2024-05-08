#!/bin/bash

#SBATCH -J llama2-13b-1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
SYSTEM="argonne-swing"
HF_NAME="meta-llama/Llama-2-13b-chat-hf"
MODEL_NAME="llama2-13b"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

module load amd-uprof
cd /home/ac.gwilkins/energy-inference/cuda/

mkdir -p ./$MODEL_NAME/$DATE/$TIME
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda-independence.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM

