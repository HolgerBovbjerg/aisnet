#!/bin/bash
#SBATCH --job-name=example_training
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (mem/cpus-per-core per cpu-core is default)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --output=example_training.out

# Default values (if not provided via sbatch argument)
CONFIG_PATH="../experiments/example/configs/"
CONFIG_NAME="config"

# Check if config path and name were passed as arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-path)
            CONFIG_PATH=$2
            shift 2
            ;;
        --config-name)
            CONFIG_NAME=$2
            shift 2
            ;;
        *)
            echo "Usage: $0 [--config-path <path>] [--config-name <name>]"
            exit 1
            ;;
    esac
done

# Output the values for debugging
echo "Using CONFIG_PATH=$CONFIG_PATH"
echo "Using CONFIG_NAME=$CONFIG_NAME"

# Calculate the total world size (the number of processes involved in the job)
# WORLD_SIZE is the product of the number of nodes and the number of tasks per node
export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# Get the hostname of the master node (the first node in the job)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Execute the job using Singularity
srun singularity exec --nv pytorch_24.03-py3.sif python -m train --config-path "${CONFIG_PATH}" --config-name "${CONFIG_NAME}"
