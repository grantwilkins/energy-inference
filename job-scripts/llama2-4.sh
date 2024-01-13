#!/bin/bash

#SBATCH -J llama2-4

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:4

cd /home/ac.gwilkins/energy-inference/llama2/
for i in {0..25}
do
	python3 llama2-cuda.py
done
