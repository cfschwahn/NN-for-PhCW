## Script to be run on cluster. MPB simulations are run one per core
# in parallel and results are logged as they are produced. 
# Caspar Schwahn August 2022

import multiprocessing as mp
import time
import csv
import numpy as np
from numpy.random import default_rng
import pandas as pd
from tqdm.auto import tqdm
import pathlib
from datetime import datetime
import sys
import os
import subprocess

### Constants and directory organisation

## directory organisation
# creates the following:
# output_dir/
#   -> output.csv  # each row contains an index, and the design parameters and FoMs of one design
#   
# log_dir/
#   -> 0.log
#   ...
#   -> (n_designs-1).log

n_designs = int(sys.argv[1])              # target number of designs to generate and evaluate
n_cpus = int(sys.argv[2])                 # number of cpu cores to use (should leave one for I/O)
out_name = sys.argv[3]                    # name of directories to be created for storing outputs
out_path = pathlib.Path(sys.argv[4])      # directory where output directory should be created
log_path = pathlib.Path(sys.argv[5])      # directory where logs directory should be created
mpb = str(sys.argv[6])                    # path to MPB program
inputFile = pathlib.Path(sys.argv[7])     # path to MPB control file 

output_dir = out_path / out_name
log_dir = log_path / (out_name + "-logs")
for d, name in zip([output_dir, log_dir], ["Output", "Logs"]):
    if d.exists():
        sys.exit("{} directory already exists.".format(name))
    else:
        d.mkdir()

#mpb = "/usr/bin/mpb"
#inputFile = "/home/nanophotgrp/Desktop/NN-for-PhCW_Caspar/code/WaveguideCTL/W1_2D_v04.ctl.txt"
#output_dir = pathlib.Path("/home/nanophotgrp/Desktop/NN-for-PhCW_Caspar/sims/Caspar/7d-no-p-shifts") / outname
#if output_dir.exists():
#    sys.exit("Output directory already exists. Delete it or choose another name.")

# design constants
cvec = {'r0': (0.2,0.45), 'r1': (0.2,0.45), 'r2': (0.2,0.45), 'r3':(0.2,0.45),
        's1': (-2.165063509461097, 0.616025),'s2': (-1.73205080756, 1.4820508),'s3': (-0.86602540378, 1.915063509),
        'p1': (0.0, 0.0), 'p2': (0.0, 0.0), 'p3': (0.0, 0.0)}
max_iter=100000
min_rad=0.2
min_sep=0.1
min_gap=min_sep

# mpb constants
n_bands = 30
n_k_points = 101

### Functions

def check_design(dvec, min_rad=0.2, min_sep=0.1, min_gap=0.1):
    """
    Returns whether or not a 10D design vector is valid i.e. within manufacturing limits.
    All values in units of a
    
    dvec (dict): design vector as a dict with keys: 
    ['r0','r1','r2','r3','s1','s2','s3','p1','p2','p3']
    :@param min_rad: minimum hole radius
    :@param min_sep: minimum hole separation
    :@param min_gap: minimum hole separation across the defect (by defaul equal to min_sep)
    
    returns (boolean): whether dvec is valid
    """
    
    # keys need to match input design vector keys as defined here               
    # r0: radius of holes in all rows other than those defined below
    ri_keys = ['r1', 'r2', 'r3', 'r0'] # ri: radius of holes in ith row from defect
    pi_keys = ['p1', 'p2', 'p3'] # shifts of the ith row parallel to the defect
    si_keys = ['s1', 's2', 's3'] # shifts of the ith row perpendicular to the defect
    
    ris = [dvec[k] for k in ri_keys]
    pis = [dvec[k] for k in pi_keys]
    sis = [dvec[k] for k in si_keys]
    
    ## Check the radii
    for ri in ris:
        # Check the lower radius limit
        if ri < min_rad:
            return False
        ## Check the upper radius limit
        if ri > (1 - min_sep)/2:
            return False
    
    ## Check periodicity of parallel row shifts
    for pi in pis:
        if abs(pi) > 0.5:
            return False
    
    ## Check geometry
    
    # 0th rows have no shifts by definition but these are added so that the 0th row can be treated 
    # the same way as any of the ith rows
    sis.append(0)
    
    n_rows = len(sis) # number of row displacements including the 0 displacement of 0th row
    default_x = [i*np.sqrt(3/4) for i in range(n_rows-1,-1,-1)] # default distance along s of each row as measured from row 0
    xs = np.array([d + s for d, s in zip(default_x, sis)]) # actual distance along s of each row in above reference frame
    
    # check that the input is not ill formed meaning that labelling rows by i retains meaning.
    # i.e. the 1st row cannot be further away from the defect than the 2nd row and so on. 
    # Ill formed but otherwise valid designs could be allowed which effectively can be used in a
    # data augmentation strategy (multiple combinations of design parameters map to the same design)
    # The periodicity could be changed similarly # TODO
    def well_formed(xs):
        # base case
        n = len(xs)
        if n == 1:
            return True
        # check positions
        for i in range(n-1):
            if xs[i] < xs[-1]:
                return False
        # recurse over the remaining rows
        return well_formed(xs[:-1])

    # run labelling check
    if well_formed(xs) is False:
        return False
    
    pis.append(0)
    
    # measure p shifts in terms of coordinates with origin on 0th row:
    default_ys = [0.5 if i%2==1 else 0 for i in range(n_rows-1,-1,-1)]

    d_matrix = np.array([ris,pis,xs,default_ys]).transpose()

    x_to_defect = n_rows*np.sqrt(3/4)
    # helper function checks one row at a time going towards the defect starting from "0th" row
    def check_rows(d_matrix):
        
        n = len(d_matrix)
        # "this" refers to the row that is being compared to all the others
        this_r, this_p, this_x, this_y_offset = d_matrix[-1]
        
        # check the minimum separation across the defect holds
        if (x_to_defect - this_x) - this_r < min_gap/2:
            return False
        
        # all rows checked
        if n == 1:
            return True
        
        # compare one row agains others
        for i in range(n-1):
            next_r, next_p, next_x, next_y_offset = d_matrix[i]
            y_offset = abs(this_y_offset-next_y_offset)
            p_diff = this_p-next_p
            smaller_y_distance = min((y_offset+p_diff)%1, 1-(y_offset+p_diff)%1)
            x_distance = next_x-this_x
            # check the minimum hole separtion:
            if np.sqrt(x_distance**2+smaller_y_distance**2)-this_r-next_r < min_sep:
                return False
        # recurse with the comparison row removed
        return check_rows(d_matrix[:-1])
    
    # return whether all constraints are respected
    return check_rows(d_matrix)

def random_design(cvec = {'r0': (0.2,0.45), 'r1': (0.2,0.45), 'r2': (0.2,0.45), 'r3':(0.2,0.45),
                          's1': (-2.165063509461097, 0.616025),
                          's2': (-1.73205080756, 1.4820508), 
                          's3': (-0.86602540378, 1.915063509),
                          'p1': (-0.5, 0.5), 'p2': (-0.5, 0.5), 'p3': (-0.5, 0.5)},
                 max_iter=10000, min_rad=0.2, min_sep=0.1, min_gap=0.1):
    """
    Samples a random design uniformly from the design space subject to manufacturing limits.
    
    :@param cvec (dict): constraint vector is a dict with structural parameter keys and values are tuples
        (min, max) which restrict the parameter to that range
        the default value covers the entire design space that is possible with 10 D and (min_rad=0.2, min_sep=0.1)
    :@max_iter (int): maximum number of invalid samples that can be generated before function stops
    
    returns dvec (dict): a valid design vector
    """
    rng = default_rng()
    for i in range(max_iter):
        dvec = {k: rng.uniform(v[0], v[1]) for k, v in cvec.items()}
        if check_design(dvec, min_rad=min_rad, min_sep=min_sep, min_gap=min_gap):
            return dvec
    print("Max iterations reached without finding a valid design")

def parse_bands(file_path, n_k_points = 101, n_bands = 30):
    """Takes a path to an MPB log file and parses the (TE) frequencies for each k-point.
    File is parsed in reverse which is necessary for some buggy log files, as noted in 
    parseObjFunctionparams in mpbParser.py by Sean Billings. Note: does not keep track of how many rows 
    inculding k-points are actually read in the file i.e. does not check for duplicates/missing information.
    Assume log files are well formed as the single core runs using cluster_generate.py appear to be have no
    duplicate k-points.
    
    :param file_path: pathlib.Path or string path to MPB log file. File assumed to exist.
    :param n_k_points: expected number of k-points (should be MPB's kinterp + 2). Note that indices start at 1 in log file
    :param n_bands: expected total number of bands. Note that indices starts at 1 in log file
    :return: band structure a matrix of (TE) frequencies indexed along first axis by band and along second axis
    by k-point (as int k label instead of acutal k value)
    """
    band_matrix = np.empty((n_bands, n_k_points)) # matrix of TE frequencies. First axis is band, second axis is k point
    
    with open(file_path) as f:
        # log files read in reverse as recommened
        for line in reversed(f.readlines()):
            # select lines containing TE freqs for each k-point (skip the header line)
            # k point is at index 1 and the 30 te frequencies start at index 6
            if "tefreqs:, " in line and "k index" not in line:
                parsed_line = line.split(",")
                k_point = int(parsed_line[1])
                te_freqs = parsed_line[6:]
                band_matrix[:, k_point-1] = te_freqs
                
    return band_matrix

def generate(index):
# helper function for a single MPB simulation. Draws random design,
# runs mpb, parses its log file, and returns the design and its bands.
    
    outputLog = log_dir / (str(index).zfill(6) + ".log")

    # choose a random design
    dvec = random_design(cvec=cvec,max_iter=max_iter, min_rad=min_rad, min_sep=min_sep, min_gap=min_gap)

    paramVectorString = ""
    for k in dvec.keys():
        paramVectorString += " " + k + "=" + str(dvec[k])

    # Note: c1, c2, and fieldFraction values taken from W1Experiment in experiment.py in backend (Sean Billings, 2015)
    p = {"c1": 4, # constant for out of plane loss,
        "c2": 220, # constant for backscattering loss,
        "calculationType": " calculation-type=4",
        "fieldFraction": 0.827, 
        "inputFile": inputFile,
        "ke": 0.5,
        "kinterp": 99, # for k resolution of 0.005
        "ks": 0.0, 
        "mpb": mpb,
        "outputFile": outputLog,
        "paramVectorString": paramVectorString,
    }

    # Call MPB as in W1Experiment in experiment.py in backend (Sean Billings, 2015)
    FNULL = open(os.devnull, 'w')

    cmd_str = str(p["mpb"]) + " Ks=" + str(p["ks"]) + " Ke=" + str(p["ke"]) + " Kinterp=" + str(p["kinterp"]) + p["calculationType"] +  p["paramVectorString"] + ' %s > %s' %(p["inputFile"], p["outputFile"])
    #print("Start", index, "running command: ", cmd_str)
    code = subprocess.run(cmd_str, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    #print("Finish: ", index, "with return code ", code.returncode, " and output", code.stdout)
    # extract bands from log file
    band_vector = parse_bands(outputLog, n_k_points=n_k_points, n_bands=30).flatten().tolist()
    #print("Parsed: ", index)
    
    return {"id":index ,"band_vector":band_vector,**dvec}


def handle_generate(n_designs=n_designs):

    # create a header for csv (with multiindexing for pandas)
    d_cols = ['r0', 'r1', 'r2', 'r3', 's1', 's2', 's3', 'p1', 'p2', 'p3']    
    top_cols = ["id"]+len(d_cols)*["design_params"]
    bot_cols = ["id"]+d_cols
    for i in range(1, n_bands+1):
        for j in range(1, n_k_points+1):
            bot_cols.append("k_{}".format(j))
            top_cols.append("band_{}".format(i))

    # set up multiprocessing
    pool = mp.Pool(n_cpus)

    # show a progress bar
    with tqdm(total=n_designs, unit="PhCW design", leave=True, dynamic_ncols=True, smoothing=0.01) as pbar:
        # create output csv
        with open(output_dir/ (out_name+".csv"), "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # write header
            writer.writerows([top_cols, bot_cols])
            
            # write results as they arrive
            # Alternatively use imap() instead of imap_unordered to guarantee order is preserved
            for result in pool.imap_unordered(generate, range(n_designs)):
                writer.writerow([result["id"]]+[result[k] for k in d_cols]+result["band_vector"])
                pbar.update()
    pbar.close()

### Dataset generation
# start timing
tik = datetime.now()

handle_generate()

#stop timing
tok = datetime.now()
print("Runtime (s): {}".format(str(tok-tik)))

