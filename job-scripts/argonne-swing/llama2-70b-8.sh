#!/bin/bash

#SBATCH -J llama2-70b-8

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:8

cd /home/ac.gwilkins/energy-inference/cuda
for i in {0..10}
do
	python3 gated-cuda.py --num_tokens 1000 --model_name meta-llama/Llama-2-70b-chat-hf
done
