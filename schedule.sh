#!/bin/sh
#PBS -N batch_DDPG
#PBS -o batch_DDPG.out
#PBS -b batch_DDPG.err
#PBS -l walltime=10:00:00
#PBS -l ncpus=32
export PYTHONPATH=$PYTHONPATH:/home/arnaud.debroissia
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
python schedule_calcs_DDPG.py &
wait
