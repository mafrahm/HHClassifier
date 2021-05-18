#!/bin/bash
#SBATCH --partition=cms-uhh,cms,allgpu
#SBATCH --time=1-00:00:00                           # Maximum time requested
#SBATCH --constraint=GPU
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --job-name  steer
#SBATCH --output    steer-%N-%j.out            # File to which STDOUT will be written
#SBATCH --error     steer-%N-%j.err            # File to which STDERR will be written
#SBATCH --mail-type ALL                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user mathis.frahm@desy.de          # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
#SBATCH --requeue

#source .setenv_V4

#source ~/.bashrc
#module load maxwell
#module load cuda
#module load anaconda/3

export PATH="/beegfs/desy/user/frahmmat/anaconda2/bin:$PATH" 
source activate condaPy27
echo $PYTHONPATH
cd /beegfs/desy/user/frahmmat/HHClassifier
python steer.py
