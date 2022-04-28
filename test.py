import argparse
import time
import yaml
import os
import ipdb
import torch
import torchvision
from shutil import rmtree
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils import utils 
from models import model_MPRnet
# from data import dataloader_pair, dataloader_pair_mix
import data as Data
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, help="")
parser.add_argument("--config_file", type=str, default='./checkpoints/config.yaml')
parser.add_argument("--phase", type=str, default='test')
args = parser.parse_args()

def write_txt(file_name, line):
    with open(file_name,'a') as log:
        log.write(line+'\n')
    print(line)

with open(args.config_file,'r') as f:
    config = yaml.load(f)
    config['is_training'] = False
    config['resume_train'] = False
    config['which_epoch'] = config['test']['which_epoch']
    utils.print_config(config)

if config['Distributed']:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")

### make saving dir
test_config = config['test']
if not os.path.exists(test_config['result_dir']):
    os.mkdir(test_config['result_dir'])
if test_config['save_dir']:
    image_save_dir = os.path.join(test_config['result_dir'],test_config['save_dir']) 
else:
    image_save_dir = os.path.join(test_config['result_dir'],config['model_name']) 

path_root = test_config['dataset']['dataroot']
# if "denoise_dataset" in path_root:
#     task_name = "denoise"
# elif "derain_dataset" in path_root:
#     task_name = "derain"
# elif "deblur_dataset" in path_root:
#     task_name = "deblur"
# elif "dehaze_dataset" in path_root:
#     task_name = "dehaze"
# elif "Mix" in path_root or "mix" in path_root:
#     task_name = "mixed"
task_name = test_config['dataset']['degrade_type']
image_save_dir = os.path.join(image_save_dir,task_name)
os.makedirs(image_save_dir,exist_ok=True)


# test
### initialize model
if config['model_class'] == "MPRnet":
    Model = model_MPRnet
else:
    raise ValueError("Model class [%s] not recognized." % config['model_class'])

model = Model.RestoreNet(config)
# model.load(config)

def test_one_dir():
    ### load datasets
    if test_config['dataset_mode'] == 'pair':
        dataset_opt = test_config['dataset']
        test_dataset = Data.create_dataset(dataset_opt, args.phase, dataset_opt['degrade_type'])
        test_dataloader = Data.create_dataloader(
            test_dataset, dataset_opt, args.phase)                         
    elif test_config['dataset_mode'] == 'mix':
        test_dataset = dataloader_pair_mix.BlurryVideo(config, train= False)
        test_dataloader = DataLoader(test_dataset,
                                        batch_size=test_config['test_batch_size'],
                                        shuffle = False)
    print(test_dataset)

    t_test_psnr = 0
    cnt = 0
    model.net_G.eval()
    record_file = os.path.join(test_config['result_dir'],test_config['save_dir'],'PSNR.txt')
    start_time = time.time()
    print('--------testing begin----------')
    for index, batch_data in enumerate(test_dataloader):
        # if index%2 == 0:
        start_time_i = time.time()
        model.set_input(batch_data)
        psnr = model.test(validation=True, multi_step=test_config['multi_step'])
            
        image_path = model.get_image_path()
        print('[time:%.3f]processing %s PSNR: %.2f'%(time.time()-start_time_i, image_path['B_path'],psnr))
        t_test_psnr += psnr
        cnt += 1

        results = model.get_current_visuals()
        utils.save_test_images(config, image_save_dir, results, image_path)
            
    message = 'Test %s model on %s [Type:%s] PSNR: %.2f'%(config['model_name'], test_config['dataset']['dataroot'],
                                                          test_config['dataset']['degrade_type'], t_test_psnr/cnt)
    print(message)
    write_txt(record_file, message)
    print('Test time %.3f'%(time.time()-start_time))


def walk_test_dir(test_dir):
    sub_dir = os.listdir(test_dir)
    if 'input' in sub_dir:
        test_one_dir()
    else:
        for dataset_i in sub_dir:
            test_config['dataset']['dataroot'] = os.path.join(test_dir,dataset_i)
            test_video_dir = os.path.join(test_dir,dataset_i)
            walk_test_dir(test_video_dir)

test_dir = test_config['dataset']['dataroot']
walk_test_dir(test_dir)


        

