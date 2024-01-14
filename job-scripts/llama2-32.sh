#!/bin/bash

#SBATCH -J llama2-32

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:8

cd /home/ac.gwilkins/energy-inference/llama2/
for i in {0..25}
do
	python3 llama2-cuda.py
done