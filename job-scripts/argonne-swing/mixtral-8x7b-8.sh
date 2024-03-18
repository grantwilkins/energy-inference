#!/bin/bash

#SBATCH -J mixtral-8x7b-8

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:8

N_NODES=1
N_GPUS_PER_NODE=8
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))
N_TOKENS=(128 256 512 1024 2048)
HF_NAME="mistralai/Mixtral-8x7B-v0.1"
MODEL_NAME="mixtral-8x7b"
BATCH_SIZES=(8 16 64 128)
SYSTEM="argonne-swing"
DATE=$(date +"%Y-%m-%d")
module load amd-uprof

cd /home/ac.gwilkins/energy-inference/cuda/

for num_tokens in "${N_TOKENS[@]}"
	do 
        AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$SYSTEM/gpus-$N_GPUS-tokens-$num_tokens-batch_size-32 python3 open-source-cuda.py --num_tokens $num_tokens --hf_name $HF_NAME --batch_size 32 --system_name $SYSTEM
    done

for batch_size in "${BATCH_SIZES[@]}"
        do 
        AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$SYSTEM/gpus-$N_GPUS-tokens-512-batch_size-$batch_size python3 open-source-cuda.py --num_tokens 512 --hf_name $HF_NAME --batch_size $batch_size --system_name $SYSTEM
    done
