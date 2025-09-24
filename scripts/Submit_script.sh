#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single node
#
#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Specify project account
#SBATCH --account=palmer-addnm
#
# No. of tasks required (max. of 40) (1 for a serial job)
#SBATCH --ntasks=40
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=168:00:00
#
# Job name
#SBATCH --job-name=scripts
#
# Output file
#SBATCH --output=./slurm_files/slurm-%j.out
#=======================================================


module purge

module load anaconda/python-3.9.7

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh 
#----------------------------------------------------------

# Modify the line below to run your program
source activate tlp_py312

#python -u /users/yhb18174/TL_project/scripts/gen_descriptors.py
#python -u /users/yhb18174/TL_project/scripts/train_embeddings.py
python -u /users/yhb18174/TL_project/scripts/train_rf_models.py

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------