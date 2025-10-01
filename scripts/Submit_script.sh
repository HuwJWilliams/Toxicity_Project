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
#SBATCH --partition=dev
#
# Specify project account
#SBATCH --account=palmer-addnm
#
# No. of tasks required (max. of 40) (1 for a serial job)
#SBATCH --ntasks=1
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=1:00:00
#
# Job name
#SBATCH --job-name=dev
#
# Output file
#SBATCH --output=./slurm_files/slurm-%j.out
#=======================================================


module purge

module load miniforge/python-3.12.10/25.3.0

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh 
#----------------------------------------------------------

# Modify the line below to run your program
mamba activate tlp_py312

# Make sure conda's libstdc++.so.6 is used instead of the system one
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#python -u /users/yhb18174/TL_project/scripts/gen_descriptors.py
#python -u /users/yhb18174/TL_project/scripts/train_embeddings.py
python -u /users/yhb18174/TL_project/scripts/test_cluster_analysis.py
echo "Worked"

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------