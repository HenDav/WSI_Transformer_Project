#!/bin/bash

# Default values
NUM_CORES=56
NUM_GPUS=4
JOB_NAME="lightning_job"

while getopts c:g:n: flag
do
    case "${flag}" in
        c) NUM_CORES=${OPTARG};;
        g) NUM_GPUS=${OPTARG};;
        n) JOB_NAME=${OPTARG};;
    esac
done

# Shift out the processed options
shift $((OPTIND -1))

# Parameters for sbatch
NUM_NODES=1
MAIL_USER="dahen@cs.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# Conda parameters
CONDA_HOME=/home/dahen/miniconda3/condabin/conda
CONDA_ENV=conda_master

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-w gipdeep10 \
	-A gipmed \
	-p gipmed \
	-o 'outputs/slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
# echo "*** Activating environment $CONDA_ENV ***"
# source $CONDA_HOME/etc/profile.d/conda.sh
# conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF
