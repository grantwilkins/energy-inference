#!/bin/bash

#SBATCH -J phi2-2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:2

cd /home/ac.gwilkins/energy-inference/cuda/
for i in {0..25}
do
	python3 open-source-cuda.py --num_tokens 1000 --model_name microsoft/phi-2
done
