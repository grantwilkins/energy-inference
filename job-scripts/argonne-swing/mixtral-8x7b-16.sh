#!/bin/bash

#SBATCH -J mixtral-16

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:8

cd /home/ac.gwilkins/energy-inference/cuda/
for i in {0..10}
do
	python3 open-source-cuda.py --num_tokens 1000 --model_name mistralai/Mixtral-8x7B-v0.1
done
