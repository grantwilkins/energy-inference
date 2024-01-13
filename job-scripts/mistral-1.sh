#!/bin/bash

#SBATCH -J mistral-1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1

cd /home/ac.gwilkins/energy-inference/mistral/
for i in {0..25}
do
	python3 mistral7b-cuda.py
done
