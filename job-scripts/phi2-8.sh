#!/bin/bash

#SBATCH -J phi2-8

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:8

cd /home/ac.gwilkins/energy-inference/phi/
for i in {0..25}
do
	python3 phi-2-cuda.py
done
