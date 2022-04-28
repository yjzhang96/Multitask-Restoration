import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import os
import torch.nn.functional as F
from . import networks, losses
from collections import OrderedDict
from utils import utils 
from .schedulers import WarmRestart,LinearDecay
from utils.image_pool import ImagePool
from ipdb import set_trace as stc
import random
class RestoreNet():
    def __init__(self, config):
        self.config = config
        
        ## configure multi-process GPU
        if config['Distributed']:
            local_rank = torch.distributed.get_rank()%len(config['gpu'])
            print('local rank:',local_rank)
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda",local_rank)
        else:
            if config['gpu']:
                device = torch.device('cuda:{}'.format(config['gpu'][0]))
                torch.cuda.set_device(device)
            else:
                device = torch.device('cpu')

        ### initial model and model parallel
        self.net_diff = networks.define_diffusion(config, device)
        # load pretrain net_diff
        if config['model']['pretrain_G'] is not None:
            pretrain_path = config['model']['pretrain_G']
            self.net_diff.restore_fn.load_state_dict(torch.load(pretrain_path))
            print('--------- Load pretrained restore model:%s ---------'%pretrain_path)
        else:
            print('--------- Train restore model from scratch ---------')
        # DDP
        if config['Distributed'] and torch.cuda.device_count()>1:
            if config['resume_train'] or not config['is_training']:
                self.load(config)
            self.net_diff.to(device)
            print("let's use", torch.cuda.device_count(),"GPUs!")
            self.net_diff = torch.nn.parallel.DistributedDataParallel(
                self.net_diff,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        # DP or CPU
        elif not config['Distributed']:
            self.net_diff.to(device)
            if len(config['gpu']) >1:
                self.net_diff = torch.nn.DataParallel(self.net_diff, config['gpu'])
            if config['resume_train'] or not config['is_training']:
                self.load(config)

        ###Loss and Optimizer
        self.MSE = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = losses.SSIMLoss('MSSSIM')
        self.criterion_char =losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss()

        if config['is_training']:
            self.optimizer_G = torch.optim.Adam( self.net_diff.parameters(), lr=config['train']['lr_G'], betas=(0.9, 0.999) )
            if config['resume_train']:
                print("------loading learning rate------")
                self.get_current_lr_from_epoch(self.optimizer_G, config['train']['lr_G'], config['start_epoch'], config['epoch'])
       

    def set_input(self,batch_data):
        # self.input = batch_data['INPUT'].to(self.device)
        self.input = batch_data['input'].cuda()
        self.target = batch_data['target'].cuda()
        self.B_path = batch_data['B_path']
        self.index = batch_data['index'].cuda()

    def optimize(self):
        degrade_num = self.config['model']['degrade_num']
        restored_list = self.net_diff.module.restore_fn(self.input, self.index)
        self.restored = restored_list[0]

        # alter_type = random.random()
        # if alter_type < 0.25:
        #     alter_index = random.randint(1,degrade_num-1)
        #     self.index = (self.index + alter_index)%degrade_num
        #     self.target = self.input

        loss_char_j = [self.criterion_char(restored_list[j],self.target) for j in range(len(restored_list))]
        self.loss_char = loss_char_j[0] + loss_char_j[1] + loss_char_j[2]
        loss_edge_j = [self.criterion_edge(restored_list[j],self.target) for j in range(len(restored_list))]
        self.loss_edge = loss_edge_j[0] + loss_edge_j[1] + loss_edge_j[2]
        self.restore_loss = (self.loss_char) + (0.05*self.loss_edge)       
        
        self.diff_loss = self.net_diff(self.input, self.restored, self.target)
        
        self.loss = 0.5 * self.restore_loss + self.diff_loss

        self.optimizer_G.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_diff.parameters(), 0.01)
        self.optimizer_G.step()

    def get_loss(self):
        
        return OrderedDict([
                            ('restore',self.restore_loss.item()),
                            ('diff',self.diff_loss.item()),
                            ('loss_tot',self.loss.item()),
                            ])
    
    @staticmethod
    def trans_func(index):
        ## denoise then end
        if index == 2: 
            return index-3
        ## derain then denoise
        if index == 1:
            return index-1
        ## deblur then end
        if index == 0:
            return index+2
    
    def diffusion_sample(self, input, restore_step, continous=False):
        if self.config['diff_sample']['sample_type'] == "generalized":
            tot_timestep = self.config['diff_sample']['n_timestep']
            if self.config['diff_sample']['skip_type'] == 'uniform':
                skip = tot_timestep // self.config['diff_sample']['sample_step']
                seq = range(0, tot_timestep, skip)
            elif self.config['diff_sample']['sample_type'] == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.tot_timestep * 0.8), self.config['sample']['sample_step']
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)] 
            # with torch.no_grad():
            if len(self.config['gpu']) > 1:
                return self.net_diff.module.skip_restore(
                    input, restore_step, seq, continous)
            else:
                return self.net_diff.skip_restore(
                    input, restore_step, seq, continous)

        elif self.config['diff_sample']['sample_type'] == "ddpm":
            # with torch.no_grad():
            if len(self.config['gpu']) > 1:
                return self.net_diff.module.restore(
                    input, restore_step, continous)
            else:
                return self.net_diff.restore(
                    input, restore_step, continous)

    def test(self, validation = False, multi_step = False):
        self.net_diff.eval()
        with torch.no_grad():
            B,C,H,W = self.target.shape
            
            output = self.input
            restore_step = self.index
            if multi_step:
                while(restore_step>=0):
                    print('degrade now:',restore_step)
                    output = self.diffusion_sample(output,restore_step)
                    # restore_step -= 1
                    restore_step = self.trans_func(restore_step)
                self.restored = output
            else:
                self.restored = self.diffusion_sample(output,restore_step)
        self.net_diff.train()
            
        # calculate PSNR
        def PSNR(img1, img2):
            MSE = self.MSE(img1,img2)
            return 10 * np.log10(1 / MSE.item())

        if validation:
            
            sharp_psnr = 0
            sharp_psnr += PSNR(self.target,self.restored) 
            return sharp_psnr

    def save(self,epoch):
        save_g_filename = 'G_net_%s.pth'%epoch
        if len(self.config['gpu'])>1:
            torch.save(self.net_diff.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        else:
            torch.save(self.net_diff.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        
        #     self.deblur_net.to(self.device)

    def load(self, config):
        load_path = os.path.join(config['checkpoints'], config['model_name'])
        load_G_file = load_path + '/' + 'G_net_%s.pth'%config['which_epoch']
        print(load_G_file) 
        if len(self.config['gpu'])>1 and isinstance(self.net_diff, nn.DataParallel):
            print('--------load model.module ----------') 
            self.net_diff.module.load_state_dict(torch.load(load_G_file))
        else:
            print('--------load model without .module ----------')
            self.net_diff.load_state_dict(torch.load(load_G_file))
        print('--------load model %s success!-------'%load_G_file)
        
        
    def schedule_lr(self, epoch,tot_epoch):
        # scheduler
        # print("current learning rate:%.7f"%self.scheduler.get_lr())
        self.get_current_lr_from_epoch(self.optimizer_G, self.config['train']['lr_G'], epoch, tot_epoch)

    def get_current_lr_from_epoch(self,optimizer, lr, epoch, tot_epoch):
        # current_lr = lr * (0.9**(epoch//decrease_step))
        current_lr = lr * (1 - epoch/tot_epoch)
        # if epoch > 500:
        #     current_lr = 0.000001
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print("current learning rate:%.7f"%(current_lr))

    def get_current_visuals(self):
        input = utils.tensor2im(self.input)
        target = utils.tensor2im(self.target)
        restored = utils.tensor2im(self.restored)
        return OrderedDict([('input',input),('target',target),('restored',restored)])
    
    def get_tensorboard_images(self):
        
        paired_train_img = OrderedDict([('input',self.input[0]),('target',self.target[0]),('restored',self.restored[0])])
        return paired_train_img

    def get_image_path(self):
        return {'B_path':self.B_path}
