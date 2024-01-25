#!/bin/bash

#SBATCH -J llama2-7b-2

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:2

cd /home/ac.gwilkins/energy-inference/cuda
for i in {0..10}
do
	python3 gated-cuda.py --num_tokens 1000 --model_name meta-llama/Llama-2-7b-chat-hf
done
