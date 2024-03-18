#!/bin/bash

#SBATCH -J llama2-13b-2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:2

N_NODES=1
N_GPUS_PER_NODE=2
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(128 256 512 1024 2048)
HF_NAME="meta-llama/Llama-2-13b-chat-hf"
MODEL_NAME="llama2-13b"
BATCH_SIZES=(8 16 64 128)
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
module load amd-uprof
cd /home/ac.gwilkins/energy-inference/cuda/

for num_tokens in "${N_TOKENS[@]}"
	do 
        AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$SYSTEM/gpus-$N_GPUS-tokens-$num_tokens-batch_size-32 python3 gated-cuda.py --num_tokens $num_tokens --hf_name $HF_NAME --batch_size 32 --system_name $SYSTEM
    done

for batch_size in "${BATCH_SIZES[@]}"
    do
        AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$SYSTEM/gpus-$N_GPUS-tokens-512-batch_size-$batch_size python3 gated-cuda.py --num_tokens 512 --hf_name $HF_NAME --batch_size $batch_size --system_name $SYSTEM
    done