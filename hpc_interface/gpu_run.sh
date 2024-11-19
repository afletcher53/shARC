#!/bin/bash

#  see https://github.com/yaacov/argparse-sh/tree/main for details
source argparse-sh/argparse.sh

define_arg "pyscript" "" "Python script to be run" "string" "true"
define_arg "cpu_mem" "82G" "Corresponds to #SBATCH --mem=" "string" "false"
define_arg "num_gpus" "1" "Number of GPUs to use" "string" "false"

parse_args "$@"

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:$num_gpus # this means requesting 1 GPU
#SBATCH --mem=$cpu_mem  # requesting enough CPU memory for data transfer - tbc if need adjusting for no. of GPUs
#SBATCH --time=15:00:00

echo $CUDA_VISIBLE_DEVICES

module load Anaconda3/2022.05

source activate shARC_venv

python $pyscript

exit 0
EOT

# FYI - https://stackoverflow.com/questions/52421068/error-in-slurm-cluster-detected-1-oom-kill-events-how-to-improve-running-jo
