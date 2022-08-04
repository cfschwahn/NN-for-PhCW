#!/bin/bash
# Slurm stuff goes here
#
#
#
#
#
#

# set variables for python script
n_designs=100
n_cpus=10
out_name='test1'
out_path='/home/nanophotgrp/PhCW2022'
log_path='/home/nanophotgrp/PhCW2022'
mpb='/usr/bin/mpb'
inputFile='/home/nanophotgrp/PhCW2022/code/WaveguideCTL/W1_2D_v04.ctl.txt'


eval "$(conda shell.bash hook)"
conda activate deep

#python --version
python cluster_generate.py $n_designs $n_cpus $out_name $out_path $log_path $mpb $inputFile

