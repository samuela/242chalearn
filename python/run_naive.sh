#!/bin/sh
#SBATCH -t 1:00:00
#SBATCH -n 16
#SBATCH --mem 64g

python naive.py 
