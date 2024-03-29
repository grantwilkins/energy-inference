#!/bin/bash

N_NODES=1
N_GPUS_PER_NODE=1
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
HF_NAME="mistralai/Mistral-7B-v0.1"
MODEL_NAME="mistral-7b"
#BATCH_SIZES=(8 16 64 128)
SYSTEM="palmetto-v100"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")


cd /home/gfwilki/energy-inference/cuda/

# for num_tokens in "${N_TOKENS[@]}"
# 	do 
#         AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 open-source-cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens $num_tokens --hf_name $HF_NAME --system_name $SYSTEM
#     done
mkdir -p ./$MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f ./$MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
python3 cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid