#!/bin/sh
#SBATCH -t 2:00:00
#SBATCH -n 16
#SBATCH --mem 32g

source venv/bin/activate
python crf.py >& crf_out.txt