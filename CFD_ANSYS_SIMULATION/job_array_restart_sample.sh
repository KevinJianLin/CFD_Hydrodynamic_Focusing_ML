#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=             #your account number from compute canada
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=50           # Specify total number cores
#SBATCH --mem-per-cpu=8G      # Specify memory per core => this command is used instead of --mem=8G ??
#SBATCH --time=02-23:00
#SBATCH --array=1-3%1        # Run a 5-job array, one job at a time.
# ---------------------------------------------------------------------



module load ansys/2021R2      # Or newer module versions
slurm_hl2hl.py --format ANSYS-FLUENT > machinefile
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
fluent 3d -t $NCORES -cnf=machinefile -mpi=intel -affinity=0 -gu -i ../ansys_journal_file/sample_array_job.jou

