import torch
import torch.nn as nn
import numpy as np
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.functional as F
import functools
from .functions import ReverseLayerF
from .UNet_discriminator import UNetDiscriminator
# from .DCN_v2.modules.modulated_deform_conv import ModulatedDeformConv_blur
from .MPRNet import MPRNet
from .MPRNet_MH import MPRNet_MH

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # m.weight.data.normal_(0.0, 0.02)
#         nn.init.xavier_uniform_(m.weight.data)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0)
#     elif classname.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='xavier', init_gain=1, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     if len(gpu_ids)>1:
    #         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net



def define_net_G(config):
    generator_name = config['model']['g_name']
    if generator_name == 'residual_net_20':
        model_g = Residual_Net(in_channel=3, out_channel=3, n_RDB=20,learn_residual=True)
    elif generator_name == 'residual_net_10':
        model_g = Residual_Net(in_channel=3, out_channel=3, n_RDB=10,learn_residual=True)
    elif generator_name == 'DMPHN':
        model_g = DMPHN_deblur()
    elif generator_name == 'MPRnet':
        model_g = MPRNet(n_feat=config['model']['n_feat'])
    elif generator_name == 'MPRnet_MH':
        model_g = MPRNet_MH(n_feat=config['model']['n_feat'])    
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    model_g = init_net(model_g,gpu_ids=config['gpu'])
    return model_g

def define_net_D(config):
    if config['model']['norm'] != None:
        norm_layer = get_norm_layer(norm_type=config['model']['norm'])
    else:
        norm_layer = None
    discriminator_name = config['model']['d_name']
    if discriminator_name == 'unet':
        model_d = UNetDiscriminator(inChannels=3, outChannels=2,use_sigmoid=config['model']['use_sigmoid'])
    
    elif discriminator_name == 'Offset':
        model_d = OffsetNet(input_nc=3,nf=16,output_nc=2)
    elif discriminator_name == 'Offset_DomClf':
        model_d = OffsetNet_with_classifier(input_nc=3,nf=16,output_nc=2)
    elif discriminator_name == 'Condition_offset':
        if norm_layer:
            model_d = OffsetNet_norm(input_nc=6,nf=16,output_nc=2, norm_layer=norm_layer)
        else:
            model_d = OffsetNet(input_nc=6,nf=16,output_nc=2, norm_layer=norm_layer)
    else:
        raise ValueError("discriminator Network [%s] not recognized." % discriminator_name)
    model_d = init_net(model_d,gpu_ids=config['gpu'])
    return model_d

def define_global_D(config, input_nc=3, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=False):        
    norm_layer = get_norm_layer(norm_type=norm)   
    patch_gan = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat=True)   
 
    # patch_gan = NLayerDiscriminator(n_layers=3,
    #                                 norm_layer=norm_layer,
    #                                 use_sigmoid=False)
    # # print(netD)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netD.cuda(gpu_ids[0])
    # netD.apply(weights_init)
    netD = init_net(patch_gan, gpu_ids=config['gpu'])
    return netD



    
############   Classes     ##############

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, use_bias, norm_layer):
        super(ResnetBlock, self).__init__()

        padAndConv_1 = [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        padAndConv_2 = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        if norm_layer:
            blocks = padAndConv_1 + [
                norm_layer(dim),
                nn.ReLU(True)
            ]  + padAndConv_2 + [
                norm_layer(dim)]
        else:
            blocks = padAndConv_1 + [
                nn.ReLU(True)
            ]  + padAndConv_2 
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        out = x + self.conv_block(x)
        return out

def TriResblock(input_nc, norm_layer=None, use_bias=True):
    Res1 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    Res2 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    Res3 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    return nn.Sequential(Res1,Res2,Res3)

def conv_TriResblock(input_nc,out_nc,stride, use_bias=True, norm_layer=None):
    Relu = nn.ReLU(True)
    if stride==1:
        pad = nn.ReflectionPad2d(2)
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=0,bias=use_bias)
    elif stride==2:
        pad = nn.ReflectionPad2d((1,2,1,2))
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=2,padding=0,bias=use_bias)
    tri_resblock = TriResblock(out_nc, norm_layer=norm_layer)
    return nn.Sequential(pad,conv,Relu,tri_resblock)
        
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=2, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)   

# Defines the PatchGAN discriminator with the specified arguments.
# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
#         super(NLayerDiscriminator, self).__init__()
#         self.use_parallel = use_parallel
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = int(np.ceil((kw-1)/2))
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]

#         nf_mult = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

#         if use_sigmoid:
#             sequence += [nn.Sigmoid()]

#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         return self.model(input)

def pixel_reshuffle(input, upscale_factor):
    """Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
	tensor of shape ``[C*r^2, H/r, W/r]``.

	See :class:`~torch.nn.PixelShuffle` for details.

	Args:
		input (Variable): Input
		upscale_factor (int): factor to increase spatial resolution by

	Examples:
		>>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
		>>> output = pixel_reshuffle(input,2)
		>>> print(output.size())
		torch.Size([1, 12, 6, 6])
	"""
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class RDB_block(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_block, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_block(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

# Residual dense block (RDB) architecture

class make_dense_plus(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_plus, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.conv2 = nn.Conv2d(growthRate, nChannels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        out = self.conv1(x)
        out = F.leaky_relu(out,negative_slope=0.2)
        out = self.conv2(out)
        out = torch.tanh(out)/2
        # import ipdb; ipdb.set_trace()
        # out = torch.cat((x, out), 1)
        out = x + out
        return out
        
class ResDenseblock_plus(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate=32):
    super(ResDenseblock_plus, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense_plus(nChannels_, growthRate))
        # nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    # self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    # import ipdb;ipdb.set_trace()
    out = self.dense_layers(x)
    # out = self.conv_1x1(out)
    out = out + x
    return out

class RDN_residual_deblur(nn.Module):
    def __init__(self):
        super(RDN_residual_deblur, self).__init__()
        self.G0 = 96
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D = 20
        self.C = 5
        self.G = 48

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(32, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 12, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # self.ResOut = ResDenseblock_plus(6,3,growthRate=32)

    def forward(self, B1, B2, F12):
        B_shuffle = pixel_reshuffle(torch.cat((B1, B2,F12), 1), 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        # residual output
        F1, F2, F3, F4 = torch.split(self.UPNet(x) + torch.cat((B1, B1, B2, B2), 1), 3, 1)
        # direct output
        # F1, F2, F3, F4 = torch.split(self.UPNet(x), 3, 1)
        
        # resdense output
        # Res_1, Res_2 = torch.split(self.UPNet(x), 6, 1)
        # # import ipdb; ipdb.set_trace()

        # F1, F2 = torch.split(self.ResOut(Res_1 + torch.cat((B1, B1), 1)), 3, 1)
        # F3, F4 = torch.split(self.ResOut(Res_2 + torch.cat((B2, B2), 1)), 3, 1)
        return F1, F2, F3, F4




class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
    # self.conv2 = nn.Conv2d(growthRate, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
  def forward(self, x):
    # import ipdb;ipdb.set_trace()
    out = self.conv1(x)
    out = F.leaky_relu(out,negative_slope=0.2)
    # out = self.conv2(out)
    out = torch.cat((x, out), 1)
    # out = x + out
    return out

# Residual dense block (RDB) architecture
class ResDenseblock(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate=32):
    super(ResDenseblock, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    # import ipdb;ipdb.set_trace()
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class Bottleneck(nn.Module):
    def __init__(self,nChannels,kernel_size=3):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=1, 
                                padding=0, bias=True)
        self.lReLU1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nChannels*2, nChannels, kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2, bias=True)
        self.lReLU2 = nn.LeakyReLU(0.2, True)
        self.model = nn.Sequential(self.conv1,self.lReLU1,self.conv2,self.lReLU2)
    def forward(self,x):
        out = self.model(x)
        return out

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x



class Residual_Net(nn.Module):
    def __init__(self, in_channel, out_channel, n_RDB,learn_residual=True):
        super(Residual_Net, self).__init__()
        self.learn_residual = learn_residual
        self.G0 = 96
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D = n_RDB
        self.C = 5
        self.G = 48

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_channel*4, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, out_channel, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, input):
        B_shuffle = pixel_reshuffle(input, 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        # residual output
        # import ipdb; ipdb.set_trace()
        x = self.UPNet(x)
        # x = torch.tanh(x)
        if self.learn_residual:
            return input + x
        else:
            return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class DMPHN_deblur(nn.Module):
    def __init__(self):
        super(DMPHN_deblur,self).__init__()
        
        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()
        self.encoder_lv4 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()
        self.decoder_lv4 = Decoder()

    def forward(self, image):
        images_lv1 = image
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
        images_lv4_1 = images_lv3_1[:,:,0:int(H/4),:]
        images_lv4_2 = images_lv3_1[:,:,int(H/4):int(H/2),:]
        images_lv4_3 = images_lv3_2[:,:,0:int(H/4),:]
        images_lv4_4 = images_lv3_2[:,:,int(H/4):int(H/2),:]
        images_lv4_5 = images_lv3_3[:,:,0:int(H/4),:]
        images_lv4_6 = images_lv3_3[:,:,int(H/4):int(H/2),:]
        images_lv4_7 = images_lv3_4[:,:,0:int(H/4),:]
        images_lv4_8 = images_lv3_4[:,:,int(H/4):int(H/2),:]

        feature_lv4_1 = self.encoder_lv4(images_lv4_1)

        feature_lv4_2 = self.encoder_lv4(images_lv4_2)
        feature_lv4_3 = self.encoder_lv4(images_lv4_3)
        feature_lv4_4 = self.encoder_lv4(images_lv4_4)
        feature_lv4_5 = self.encoder_lv4(images_lv4_5)
        feature_lv4_6 = self.encoder_lv4(images_lv4_6)
        feature_lv4_7 = self.encoder_lv4(images_lv4_7)
        feature_lv4_8 = self.encoder_lv4(images_lv4_8)
        feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        residual_lv4_top_left = self.decoder_lv4(feature_lv4_top_left)
        residual_lv4_top_right = self.decoder_lv4(feature_lv4_top_right)
        residual_lv4_bot_left = self.decoder_lv4(feature_lv4_bot_left)
        residual_lv4_bot_right = self.decoder_lv4(feature_lv4_bot_right)

        feature_lv3_1 = self.encoder_lv3(images_lv3_1 + residual_lv4_top_left)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2 + residual_lv4_top_right)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4 + residual_lv4_bot_right)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = self.decoder_lv2(feature_lv2)

        feature_lv1 = self.encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
        deblur_image = self.decoder_lv1(feature_lv1)
        return deblur_image