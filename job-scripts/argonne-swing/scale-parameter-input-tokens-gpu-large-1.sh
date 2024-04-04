#!/bin/bash

#SBATCH -J scale-parameter-input-tokens-gpu-large

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=gpu-large
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
SYSTEM="argonne-swing-gpu-large"
module load amd-uprof

HF_NAME="tiiuae/falcon-40b"
MODEL_NAME="falcon-40b"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p $MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f $MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid

HF_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="llama2-7b"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p $MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f $MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid

HF_NAME="meta-llama/Llama-2-13b-chat-hf"
MODEL_NAME="llama2-13b"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
