#!/bin/sh
#PBS -N batch_CMA
#PBS -o batch_CMA.out
#PBS -b batch_CMA.err
#PBS -l walltime=10:00:00
#PBS -l ncpus=32
export PYTHONPATH=$PYTHONPATH:/home/arnaud.debroissia
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
python cma_mc.py &
wait
