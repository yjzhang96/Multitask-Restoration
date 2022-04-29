import argparse
import time
import os
import ipdb
import yaml
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import numpy as np 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



import data as Data
from models import model_MPRnet, model_Diff_MPR
from utils import utils 
from tensorboardX import SummaryWriter
# torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./checkpoints/config.yaml')
parser.add_argument("--local_rank", type=int, help="")
parser.add_argument("--phase", type=str, default='train')
args = parser.parse_args()

with open(args.config_file,'r') as f:
    config = yaml.load(f)
    utils.print_config(config)

if config['Distributed']:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")

### make saving dir
if not os.path.exists(config['checkpoints']):
    os.makedirs(config['checkpoints'],exist_ok=True)
model_save_dir = os.path.join(config['checkpoints'],config['model_name']) 
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir,exist_ok=True)
os.system('cp %s %s'%(args.config_file, model_save_dir))



### load datasets
if config['dataset_mode'] == 'pair':
    for phase, dataset_opt in config['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_blur_set = Data.create_dataset(dataset_opt, phase, 'blur')
            train_blur_loader = Data.create_dataloader(
                train_blur_set, dataset_opt, phase)
            train_rain_set = Data.create_dataset(dataset_opt, phase, 'rain')
            train_rain_loader = Data.create_dataloader(
                train_rain_set, dataset_opt, phase)
            train_noise_set = Data.create_dataset(dataset_opt, phase, 'noise')
            train_noise_loader = Data.create_dataloader(
                train_noise_set, dataset_opt, phase)
            train_light_set = Data.create_dataset(dataset_opt, phase, 'lowlight')
            train_light_loader = Data.create_dataloader(
                train_light_set, dataset_opt, phase)
            train_degrade_num = dataset_opt['degrade_num']
        elif phase == 'val':
            val_blur_set = Data.create_dataset(dataset_opt, phase, 'blur')
            val_blur_loader = Data.create_dataloader(
                val_blur_set, dataset_opt, phase)
            val_rain_set = Data.create_dataset(dataset_opt, phase, 'rain')
            val_rain_loader = Data.create_dataloader(
                val_rain_set, dataset_opt, phase)
            val_noise_set = Data.create_dataset(dataset_opt, phase, 'noise')
            val_noise_loader = Data.create_dataloader(
                val_noise_set, dataset_opt, phase)
            val_light_set = Data.create_dataset(dataset_opt, phase, 'lowlight')
            val_light_loader = Data.create_dataloader(
                val_light_set, dataset_opt, phase)    
elif config['dataset_mode'] == 'mix':
    train_dataset = Data.dataloader_pair_mix.BlurryVideo(config, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['batch_size'],
                                    sampler=DistributedSampler(train_dataset),
                                    num_workers=16)
    val_dataset = Data.dataloader_pair_mix.BlurryVideo(config, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['val']['val_batch_size'],
                                    sampler=DistributedSampler(val_dataset))
else:
    raise ValueError("dataset_mode [%s] not recognized." % config['dataset_mode'])
# print("train_dataset:",train_dataset)
# print("val_dataset",val_dataset)


### initialize model
if config['model_class'] == "MPRNet":
    Model = model_MPRnet
    os.system('cp %s %s'%('models/model_MPRnet.py', model_save_dir))
    os.system('cp models/%s.py %s'%(config['model']['g_name'], model_save_dir)) 
if config['model_class'] == "Diff_MPR":
    Model = model_Diff_MPR
    os.system('cp %s %s'%('models/model_Diff_MPR.py', model_save_dir))
    os.system('cp models/%s.py %s'%(config['model']['g_name'], model_save_dir)) 
    os.system('cp models/diffusion.py %s'%(model_save_dir)) 
else:
    raise ValueError("Model class [%s] not recognized." % config['model_class'])




###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def write_lr():
    pass


###Initializing VGG16 model for perceptual loss
# vgg16 = torchvision.models.vgg16(pretrained=True)
# vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
# vgg16_conv_4_3.to(device)
# for param in vgg16_conv_4_3.parameters():
# 		param.requires_grad = False



### Initialization
if config['resume_train']:
    assert config['start_epoch'] != 0
else:
    os.system('rm %s/%s/psnr_log.txt'%(config['checkpoints'], config['model_name']))
    # os.system('rm %s/%s/loss_log.txt'%(config['checkpoints'], config['model_name']))
    os.system('rm %s/%s/event*'%(config['checkpoints'], config['model_name']))


# init model
model = Model.RestoreNet(config)

# tensorboard writter
writer = SummaryWriter(model_save_dir)

def display_loss(loss,epoch,tot_epoch,step,step_per_epoch,time):
    loss_writer = ""
    for key, value in loss.items():
        loss_writer += "%s:%.4f\t"%(key,value)
    messege = "epoch[%d/%d],step[%d/%d],time[%.3fs]:%s"%(epoch,tot_epoch,step,step_per_epoch,time,loss_writer)
    print(messege)
    log_name = os.path.join(config['checkpoints'],config['model_name'],'loss_log.txt')   
    with open(log_name,'a') as log:
        log.write(messege+'\n')

# validation
def validation_pair(iter):
    t_b_psnr = 0
    psnr_tot = 0
    cnt = 0
    start_time = time.time()
    print('--------validation begin----------')
    for index, batch_data in enumerate(val_blur_loader):
        model.set_input(batch_data)
        psnr_per = model.test(validation=True)
        psnr_tot += psnr_per
        cnt += 1
        if index > 100:
            break
    message = 'Iteration %s restore PSNR: %.2f \n'%(iter, psnr_tot/cnt)
    print(message)
    # print('using time %.3f'%(time.time()-start_time))
    log_name = os.path.join(config['checkpoints'],config['model_name'],'psnr_log.txt')   
    with open(log_name,'a') as log:
        log.write(message)
    return psnr_tot/cnt

# training
val_restore_psnr = validation_pair(config['start_epoch'])
writer.add_scalar('PairPSNR/restore', val_restore_psnr, config['start_epoch'])

results = model.get_current_visuals()
utils.save_train_sample(config, 0, results)
best_psnr = 0.0
total_iter = 0
for epoch in range(config['start_epoch'], config['epoch']):
    epoch_start_time = time.time()
    iter_per_epoch = max(len(train_blur_loader),len(train_rain_loader),len(train_noise_loader))
    print('There is {:d} iteration in one epoch:'.format(iter_per_epoch))
    blur_data_iter = iter(train_blur_loader)
    rain_data_iter = iter(train_rain_loader)
    noise_data_iter = iter(train_noise_loader)
    light_data_iter = iter(train_light_loader)
    for step in range(iter_per_epoch):
        total_iter += 1
        # # training step 2
        time_step1 = time.time()
        random_type = random.randint(0,train_degrade_num-1)
        if random_type == 0:
            try:
                train_data = next(blur_data_iter)
            except:
                blur_data_iter = iter(train_blur_loader)
                continue
        elif random_type == 1:
            try:
                train_data = next(rain_data_iter)
            except:
                rain_data_iter = iter(train_rain_loader)
                continue 
        elif random_type == 2:
            try:
                train_data = next(noise_data_iter) 
            except:
                print("The end of noise data iterator")
                noise_data_iter = iter(train_noise_loader)
                continue
        elif random_type == 3:
            try:
                train_data = next(light_data_iter) 
            except:
                print("The end of light data iterator")
                light_data_iter = iter(train_light_loader)
                continue
        else:
            raise TypeError('dataloader type not recognized')

        model.set_input(train_data)        
        model.optimize()     
            

        if (step+1)%config['display_freq'] == 0:
            #print a sample result in checkpoints/model_name/samples
            loss = model.get_loss()
            time_ave = (time.time() - time_step1)/config['display_freq']
            display_loss(loss,epoch,config['epoch'],step,iter_per_epoch,time_ave)

            results = model.get_current_visuals()
            utils.save_train_sample(config, step, results)
            
            for key, value in loss.items():
                writer.add_scalar(key,value,iter_per_epoch*epoch+step)


        if total_iter % config['save_iter'] == 0:
            # model.save(total_iter)
            print('End of Iteration [%d/%d] \t Time Taken: %d sec' % (total_iter, iter_per_epoch, time.time() - epoch_start_time))
        # paired_results = model.get_tensorboard_images()
        # writer.add_image('Pair/input', paired_results['input'],epoch)
        # writer.add_image('Pair/target', paired_results['target'],epoch)
        # writer.add_image('Pair/restored', paired_results['restored'],epoch)
        

        if total_iter %config['val_freq'] == 0:
            val_restore_psnr  = validation_pair(total_iter)
            writer.add_scalar('PairPSNR/restore', val_restore_psnr, total_iter)
            results = model.get_current_visuals()
            utils.save_train_sample(config, step, results)

            if val_restore_psnr > best_psnr:
                best_psnr = val_restore_psnr
                model.save('best')

    # schedule learning rate
    model.schedule_lr(epoch,config['epoch'])
    model.save('latest')
    print('End of epoch [%d/%d] \t Time Taken: %d sec' % (epoch, config['epoch'], time.time() - epoch_start_time))
