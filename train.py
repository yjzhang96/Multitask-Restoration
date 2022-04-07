import argparse
import time
import os
import ipdb
import yaml
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
import torch.distributed as dist
dist.init_process_group(backend="nccl")

from data import dataloader_pair,dataloader_pair_mix
from models import model_MPRnet
from utils import utils 
from tensorboardX import SummaryWriter
# torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./checkpoints/config.yaml')
parser.add_argument("--local_rank", type=int, help="")
args = parser.parse_args()

with open(args.config_file,'r') as f:
    config = yaml.load(f)
    for key,value in config.items():
        print('%s:%s'%(key,value))

### make saving dir
if not os.path.exists(config['checkpoints']):
    os.mkdir(config['checkpoints'])
model_save_dir = os.path.join(config['checkpoints'],config['model_name']) 
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
os.system('cp %s %s'%(args.config_file, model_save_dir))



### load datasets
if config['dataset_mode'] == 'pair':
    train_dataset = dataloader_pair.BlurryVideo(config, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle = True,
                                    num_workers=16)
    val_dataset = dataloader_pair.BlurryVideo(config, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['val']['val_batch_size'],
                                    shuffle = True)
elif config['dataset_mode'] == 'mix':
    train_dataset = dataloader_pair_mix.BlurryVideo(config, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['batch_size'],
                                    sampler=DistributedSampler(train_dataset),
                                    num_workers=16)
    val_dataset = dataloader_pair_mix.BlurryVideo(config, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['val']['val_batch_size'],
                                    sampler=DistributedSampler(val_dataset))
else:
    raise ValueError("dataset_mode [%s] not recognized." % config['dataset_mode'])
print("train_dataset:",train_dataset)
print("val_dataset",val_dataset)


### initialize model
if config['model_class'] == "MPRnet":
    Model = model_MPRnet
    os.system('cp %s %s'%('models/model_MPRnet.py', model_save_dir))

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
    if config['which_epoch'] != 'latest':
        config['start_epoch'] = int(config['which_epoch'])
    elif config['which_epoch'] == 'latest':
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
def validation_pair(epoch):
    t_b_psnr = 0
    psnr_tot = 0
    cnt = 0
    start_time = time.time()
    print('--------validation begin----------')
    for index, batch_data in enumerate(val_dataloader):
        model.set_input(batch_data)
        psnr_per = model.test(validation=True)
        psnr_tot += psnr_per
        cnt += 1
        if index > 100:
            break
    message = 'Pair-data epoch %s restore PSNR: %.2f \n'%(epoch, psnr_tot/cnt)
    print(message)
    print('using time %.3f'%(time.time()-start_time))
    log_name = os.path.join(config['checkpoints'],config['model_name'],'psnr_log.txt')   
    with open(log_name,'a') as log:
        log.write(message)
    return psnr_tot/cnt

# training
val_restore_psnr = validation_pair(config['start_epoch'])
writer.add_scalar('PairPSNR/restore', val_restore_psnr, config['start_epoch'])

best_psnr = 0.0
for epoch in range(config['start_epoch'], config['epoch']):
    epoch_start_time = time.time()
    step_per_epoch = len(train_dataloader)
    # for step, (batch_data1, batch_data2) in enumerate(zip(train_dataloader_gt,train_dataloader_unpair)):
    G_iter = 0
    D_iter = 0
    for step, batch_data in enumerate(train_dataloader):
        p = float(step + epoch * step_per_epoch) / config['epoch'] / step_per_epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # # training step 2
        time_step1 = time.time()

        model.set_input(batch_data)        
        model.optimize()     
            

        if step%config['display_freq'] == 0:
            #print a sample result in checkpoints/model_name/samples
            loss = model.get_loss()
            time_ave = (time.time() - time_step1)/config['display_freq']
            display_loss(loss,epoch,config['epoch'],step,step_per_epoch,time_ave)

            results = model.get_current_visuals()
            utils.save_train_sample(config, epoch, results)
            
            for key, value in loss.items():
                writer.add_scalar(key,value,step_per_epoch*epoch+step)

    # schedule learning rate
    model.schedule_lr(epoch,config['epoch'])
    model.save('latest')
    print('End of epoch [%d/%d] \t Time Taken: %d sec' % (epoch, config['epoch'], time.time() - epoch_start_time))

    if epoch%config['save_epoch'] == 0:
        model.save(epoch)
    # paired_results = model.get_tensorboard_images()
    # writer.add_image('Pair/input', paired_results['input'],epoch)
    # writer.add_image('Pair/target', paired_results['target'],epoch)
    # writer.add_image('Pair/restored', paired_results['restored'],epoch)
    

    if epoch%config['val_freq'] == 0:
        val_restore_psnr  = validation_pair(epoch)
        writer.add_scalar('PairPSNR/restore', val_restore_psnr, epoch)

        if val_restore_psnr > best_psnr:
            best_psnr = val_restore_psnr
            model.save('best')

