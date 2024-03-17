#!/bin/bash

#SBATCH -J mistral-1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

cd /home/ac.gwilkins/energy-inference/cuda/
python3 open-source-cuda.py --num_tokens 128 --hf_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 256 --hf_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 1024 --hf_name mistralai/Mistral-7B-v0.1
python3 open-source-cuda.py --num_tokens 2048 --hf_name mistralai/Mistral-7B-v0.1

python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 --batch_size 8
python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 --batch_size 16
python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 --batch_size 32
python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 --batch_size 64
python3 open-source-cuda.py --num_tokens 512 --hf_name mistralai/Mistral-7B-v0.1 --batch_size 128