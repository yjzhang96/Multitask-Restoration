import os
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, help="")
parser.add_argument("--out_dir", type=str)
args = parser.parse_args()

result_dir = args.result_dir
dataset_names = os.listdir(result_dir)

for data_i in dataset_names:
    if os.path.isdir(os.path.join(result_dir,data_i)):
        input_dir = os.path.join(result_dir,data_i,'input')
        target_dir = os.path.join(result_dir,data_i,'target')
        os.makedirs(input_dir,exist_ok=True)
        os.makedirs(target_dir,exist_ok=True)

        files = os.listdir(os.path.join(result_dir,data_i))
        input_files = sorted(glob.glob(os.path.join(result_dir,data_i,'*_restored.png')))
        target_files = sorted(glob.glob(os.path.join(result_dir,data_i,'*_target.png')))
        for j in range(len(input_files)):
            old_input_name = os.path.basename(input_files[j])
            new_input_path = os.path.join(result_dir,data_i,'input',old_input_name)
            old_target_name = os.path.basename(target_files[j])
            new_target_path = os.path.join(result_dir,data_i,'target',old_target_name)
            print(input_files[j],new_input_path)
            os.system('mv %s %s'%(input_files[j],new_input_path))
            os.system('mv %s %s'%(target_files[j],new_target_path))