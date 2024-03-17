#!/bin/bash

#SBATCH -J mistral-2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:2

cd /home/ac.gwilkins/energy-inference/cuda/
python3 open-source-cuda.py --num_tokens 128 --model_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 256 --model_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 
python3 open-source-cuda.py --num_tokens 1024 --model_name mistralai/Mistral-7B-v0.1
python3 open-source-cuda.py --num_tokens 2048 --model_name mistralai/Mistral-7B-v0.1

python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 --batch_size 8
python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 --batch_size 16
python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 --batch_size 32
python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 --batch_size 64
python3 open-source-cuda.py --num_tokens 512 --model_name mistralai/Mistral-7B-v0.1 --batch_size 128