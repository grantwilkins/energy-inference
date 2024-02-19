#!/bin/bash
#SBATCH --job-name=alpa_multinode_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1

# Get names of nodes assigned
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# By default, set the first node to be head_node on which we run HEAD of Ray
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Setup port and variables needed
gpus_per_node=1
port=20000
ip_head=$head_node_ip:$port
export ip_head
# Start HEAD in background of head node
srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $gpus_per_node --block &

# Optional, sometimes needed for old Ray versions
sleep 10

# Start worker nodes
# Number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
# Iterate on each node other than head node, start ray worker and connect to HEAD
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus $gpus_per_node --block &
    sleep 5
done

# Run Alpa test
python3 ../../cuda/open-source-cuda-ray.py

# Optional. Slurm will terminate all processes automatically
ray stop
conda deactivate
exit