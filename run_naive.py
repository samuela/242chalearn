#!/bin/sh
#SBATCH -t 1:00:00
#SBATCH -n 32
#SBATCH --mem 32g

python naive.py 
