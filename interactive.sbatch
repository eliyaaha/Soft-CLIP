#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition main
#SBATCH --qos normal

#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gpus=0
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -o /home/omertole/AIMD final project/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

echo Sat 06 Jun 2026 19:49:26 IDT
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
while true; do
        eval $launch
done
