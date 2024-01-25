#!/bin/bash

#SBATCH -J mixtral-4

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:4

cd /home/ac.gwilkins/energy-inference/cuda/
for i in {0..10}
do
	python3 open-source-cuda.py --num_tokens 1000 --model_name mistralai/Mixtral-8x7B-v0.1
done
