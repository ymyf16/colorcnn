#!/bin/bash
#SBATCH --job-name=colorcnn_reorg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x_%j.out    # %x = job-name, %j = job-ID
#SBATCH --error=logs/%x_%j.err     # separate STDERR log
#SBATCH --mem=100G 
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

# make sure the logs directory exists
mkdir -p logs

# load your environment
source /opt/linux/rocky/8/software/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate colorcnn


echo "[$(date)] Starting job $SLURM_JOB_NAME ($SLURM_JOB_ID) task $SLURM_ARRAY_TASK_ID"

# run your Python script; STDOUT/ERR goes into the .out/.err files
LAYERS=(0 2 5 7)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

python color_cnn_script_reorg.py --layer $LAYER

echo "[$(date)] Finished job $SLURM_JOB_NAME ($SLURM_JOB_ID) task $SLURM_ARRAY_TASK_ID"