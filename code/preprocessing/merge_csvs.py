## script to merge multiple csvs from simulation batches into one
# Caspar Schwahn, August 2022

import numpy as np
import pandas as pd
import pathlib

def merge_csvs(out_name, path_list, out_dir):
    
    out_dir = pathlib.Path(out_dir)
    out_path = out_dir / out_name
    if out_path.exists():
        print("Output file name already exists. Delete existing file or rename")
        return

    df_list = []

    def helper(f_path):
        df = pd.read_csv(f_path, header = [0,1])
        df_list.append(df)
        print(f"Added: {f_path}")
        
    for p in path_list:
        path = pathlib.Path(p)
        if not path.exists():
            print(f"File/directory not found: {path}")
            return
        else:
            if path.is_dir():
                for r in path.rglob("*.csv"):
                    helper(r)
            else:
                helper(path)

    out_df = pd.concat(df_list, ignore_index=True)
    out_df = out_df[out_df.columns.drop("id")]
    out_df.to_csv(out_path, header=True, index=False)
    print(f"Merged csv written to {out_path}")

if __name__ == "__main__":
    
    ''''
    out_name = "combined_101_5_unsorted.csv"
    path_list = ["/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/batches"]
    out_dir = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/combined"
    '''
    '''
    out_name = "combined_101_5_parity_sorted.csv"
    path_list = ["/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/101k-30b-parity/batches"]
    out_dir = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/101k-30b-parity/combined"    
    '''
    out_name = "combined_101_30_parity_sorted_3d.csv"
    path_list = ["/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/gen-3d/batches"]
    out_dir =  "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/gen-3d/combined" 
    
    merge_csvs(out_name, path_list, out_dir)
    
