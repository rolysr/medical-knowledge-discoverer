#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="ML_GPU"

#SBATCH --time=2-00:00:00

#SBATCH --ntasks=1


#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100000M

#SBATCH --partition=graphic
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:A100:1


########## MODULES ##########

set -e
module purge
module load cuda/11.4
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


########## PATHS ##########

#Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# project path
project="$scratch/medical-knowledge-discoverer"

# Script output
output="$project/output"
mkdir -p $output

# save path
save="/data/tsa/destevez/dennis/ML/$SLURM_JOB_ID"
mkdir -p $save


########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ML/medical-knowledge-discoverer/* $project
echo "end of copy"


######### RUN ##########

cd $project
echo "runing............"
srun python3 $project/t5.py
echo "end of run"


########## SAVE ##########

echo "saving............"
cp -r $output $save
echo "end of save"


########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########