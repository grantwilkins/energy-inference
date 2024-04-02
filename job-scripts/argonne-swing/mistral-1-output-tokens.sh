#!/bin/bash

#SBATCH -J mistral-7b-1-output-tokens

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(8 16 32 64 128 256 512 1024 2048 4096)
HF_NAME="mistralai/Mistral-7B-v0.1"
MODEL_NAME="mistral-7b"
#BATCH_SIZES=(8 16 64 128)
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
module load amd-uprof

cd /home/ac.gwilkins/energy-inference/cuda/

mkdir -p $MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f $MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda-output-tokens-test.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid  
