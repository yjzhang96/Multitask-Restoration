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
        # DDP
        self.net_G = networks.define_net_G(config)
        if config['Distributed'] and torch.cuda.device_count()>1:
            if config['resume_train'] or not config['is_training']:
                self.load(config)
            self.net_G.to(device)
            print("let's use", torch.cuda.device_count(),"GPUs!")
            self.net_G = torch.nn.parallel.DistributedDataParallel(
                self.net_G,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        # DP or CPU
        elif not config['Distributed']:
            self.net_G.to(device)
            if len(config['gpu']) >1:
                self.net_G = torch.nn.DataParallel(self.net_G, config['gpu'])
            if config['resume_train'] or not config['is_training']:
                self.load(config)

        ###Loss and Optimizer
        self.MSE = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = losses.SSIMLoss('MSSSIM')
        self.criterion_char =losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss()

        self.degrade_num = self.config['model']['degrade_num'] 

        if config['is_training']:
            self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=config['train']['lr_G'], betas=(0.9, 0.999) )
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

        # alter_type = random.random()
        # if alter_type < 0.25:
        #     alter_index = random.randint(1,degrade_num-1)
        #     self.index = (self.index + alter_index)%degrade_num
        #     self.target = self.input
        B = self.index.shape
        index_now = torch.Tensor([degrade_num -1]).repeat(B).cuda()
        output = self.input
        self.loss = 0.0
        while (index_now >= self.index):
            if index_now == self.index:
                restored_list = self.net_G(output, index_now)
                ## degrade type match index_now
                loss_char_j = [self.criterion_char(restored_list[j],self.target) for j in range(len(restored_list))]
                self.loss_char = loss_char_j[0] + loss_char_j[1] + loss_char_j[2]
                loss_edge_j = [self.criterion_edge(restored_list[j],self.target) for j in range(len(restored_list))]
                self.loss_edge = loss_edge_j[0] + loss_edge_j[1] + loss_edge_j[2]
                self.loss += (self.loss_char) + (0.05*self.loss_edge)       
            elif index_now > self.index:
                ## degrade type do not match index_now
                with torch.no_grad():
                    restored_list = self.net_G(output, index_now)
                # loss_char_j = [self.criterion_char(restored_list[j],self.input) for j in range(len(restored_list))]
                # self.loss_char = loss_char_j[0] + loss_char_j[1] + loss_char_j[2]
                # loss_edge_j = [self.criterion_edge(restored_list[j],self.input) for j in range(len(restored_list))]
                # self.loss_edge = loss_edge_j[0] + loss_edge_j[1] + loss_edge_j[2]
                # self.loss += 1 * ((self.loss_char) + (0.05*self.loss_edge))
            index_now -= 1
            output = restored_list[0]

        self.restored = restored_list[0]
        self.optimizer_G.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), 0.01)
        self.optimizer_G.step()

    def get_loss(self):
        
        return OrderedDict([
                            ('char',self.loss_char.item()),
                            ('edge',self.loss_edge.item()),
                            ('loss_tot',self.loss.item()),
                            ])
    
    @staticmethod
    def trans_func(index):
        ## loghtlight
        if index == 3:
            return index-1
        ## denoise then end
        if index == 2: 
            return index-1
        ## derain then denoise
        if index == 1:
            return index-1
        ## deblur then end
        if index == 0:
            return index-1
        

    def test(self, validation = False, multi_step = True, continous=True):
        with torch.no_grad():
            B,C,H,W = self.target.shape
            
            output = self.input
            ret_output = self.input
            restore_step = self.index
    
            index_now = torch.Tensor([self.degrade_num -1]).repeat(B).cuda()
            if multi_step:
                while(index_now>=self.index):
                    print('degrade now:',index_now)
                    restore_list = self.net_G(output,index_now)
                    output = restore_list[0]
                    # restore_step -= 1
                    index_now = self.trans_func(index_now)
                    ret_output = torch.cat([ret_output, output],dim=0)
                
                if continous:
                    self.restored = ret_output
                else:
                    self.restored = output
            else:
                restore_list = self.net_G(output,restore_step)
                self.restored = restore_list[0]

            
        # calculate PSNR
        def PSNR(img1, img2):
            if len(img2) >1:
                print('Warning: restore img shape:%s, target img shape: %s'%(img2.shape, img1.shape))
                img2 = img2[-1]
            MSE = self.MSE(img1,img2)
            return 10 * np.log10(1 / MSE.item())

        if validation:
            
            sharp_psnr = 0
            sharp_psnr += PSNR(self.target,self.restored) 
            return sharp_psnr

    def save(self,epoch):
        save_g_filename = 'G_net_%s.pth'%epoch
        if len(self.config['gpu'])>1:
            torch.save(self.net_G.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        else:
            torch.save(self.net_G.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        
        #     self.deblur_net.to(self.device)

    def load(self, config):
        load_path = os.path.join(config['checkpoints'], config['model_name'])
        load_G_file = load_path + '/' + 'G_net_%s.pth'%config['which_epoch']
        print(load_G_file) 
        if len(self.config['gpu'])>1 and isinstance(self.net_G, nn.DataParallel):
            print('--------load model.module ----------') 
            self.net_G.module.load_state_dict(torch.load(load_G_file))
        else:
            print('--------load model without .module ----------')
            self.net_G.load_state_dict(torch.load(load_G_file, map_location='cpu'))
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
