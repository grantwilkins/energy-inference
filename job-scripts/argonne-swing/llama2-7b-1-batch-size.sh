#!/bin/bash

#SBATCH -J llama2-7b-1-independence

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(128 256 512 1024 2048 4096)
HF_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="llama2-7b"
#BATCH_SIZES=(8 16 64 128)
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
module load amd-uprof

cd /home/ac.gwilkins/energy-inference/cuda/

# for num_tokens in "${N_TOKENS[@]}"
# 	do 
#         AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 open-source-cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens $num_tokens --hf_name $HF_NAME --system_name $SYSTEM
#     done

AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda-batch-size.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
