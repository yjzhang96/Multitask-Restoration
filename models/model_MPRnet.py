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

class RestoreNet():
    def __init__(self, config):
        self.config = config
        if config['gpu']:
            self.device = torch.device('cuda:{}'.format(config['gpu'][0]))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        ### initial model
        self.net_G = networks.define_net_G(config)
        
        ###Loss and Optimizer
        self.MSE = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = losses.SSIMLoss('MSSSIM')
        self.criterion_char =losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss()

        if config['is_training']:
            self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=config['train']['lr_G'], betas=(0.9, 0.999) )
            
        if config['resume_train']:
            print("------loading learning rate------")
    
            self.get_current_lr_from_epoch(self.optimizer_G, config['train']['lr_G'], config['start_epoch'], config['epoch'])


    def set_input(self,batch_data):
        # self.input = batch_data['INPUT'].to(self.device)
        self.input = batch_data['input'].cuda()
        if batch_data['gt'][0]:
            self.target_exist = True
            self.target = batch_data['target'].cuda()
        else:
            self.target_exist = False
            self.target = batch_data['target'].cuda()
        self.B_path = batch_data['B_path']


    def optimize(self):
        self.restored = self.net_G(self.input)

        
        loss_char_j = [self.criterion_char(self.restored[j],self.target) for j in range(len(self.restored))]
        self.loss_char = loss_char_j[0] + loss_char_j[1] + loss_char_j[2]
        loss_edge_j = [self.criterion_edge(self.restored[j],self.target) for j in range(len(self.restored))]
        self.loss_edge = loss_edge_j[0] + loss_edge_j[1] + loss_edge_j[2]
        self.loss = (self.loss_char) + (0.05*self.loss_edge)       
        
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
    
    def test(self, validation = False):
        with torch.no_grad():
            B,C,H,W = self.target.shape
            self.restored = self.net_G(self.input)
            
            
        # calculate PSNR
        def PSNR(img1, img2):
            MSE = self.MSE(img1,img2)
            return 10 * np.log10(1 / MSE.item())

        if validation:
            
            sharp_psnr = 0
            sharp_psnr += PSNR(self.target,self.restored[0]) 
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
        
        if len(self.config['gpu'])>1:
            self.net_G.module.load_state_dict(torch.load(load_G_file))
        else:
            self.net_G.load_state_dict(torch.load(load_G_file))
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
        restored = utils.tensor2im(self.restored[0])
        return OrderedDict([('input',input),('target',target),('restored',restored)])
    
    def get_tensorboard_images(self):
        
        paired_train_img = OrderedDict([('input',self.input[0]),('target',self.target[0]),('restored',self.restored[0][0])])
        return paired_train_img

    def get_image_path(self):
        return {'B_path':self.B_path}
