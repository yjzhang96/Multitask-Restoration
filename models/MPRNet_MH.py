"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import math

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, type_index):
        count = self.dim // 2
        step = torch.arange(count, dtype=type_index.dtype,
                            device=type_index.device) / count
        encoding = type_index.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.type_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, type_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.type_func(type_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.type_func(type_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
## Degradation Aware Layer
class DALayer(nn.Module):
    def __init__(self, channel, type_imb_dim, num_head=4, reduction=4, bias=False):
        super(DALayer, self).__init__()
        self.num_head = num_head
        ## multi-head featrue: channel -> num_head * channel
        self.conv_mh = nn.Conv2d(channel, channel*num_head, 1, padding=0, bias=bias)
        self.conv_sig = nn.Sequential(
                nn.Conv2d(type_imb_dim, type_imb_dim // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(type_imb_dim // reduction, num_head, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
        # multi-tail
        self.conv_mt = nn.Conv2d(channel*num_head, channel, 1, padding=0, bias=bias)

    def forward(self, x, index_emb):
        B,C,H,W = x.shape
        y = self.conv_mh(x)
        dg_attn = self.conv_sig(index_emb.view(B,-1,1,1))
        out = y.view(B,self.num_head,C,H,W) * dg_attn.view(B,self.num_head,1,1,1)
        out = self.conv_mt(out.view(B,self.num_head*C,H,W))
        # torch.sum(out, dim=1)
        return out

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, type_emb_dim=None, use_affine_level=False):
        super(CAB, self).__init__()
        
        # if type_emb_dim:
        #     self.type_func = FeatureWiseAffine(type_emb_dim,n_feat, use_affine_level)
        # else:
        #     self.type_func = None

        # modules_body = []
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1 = act
        if type_emb_dim:
            self.DA = DALayer(n_feat, type_emb_dim, bias=bias)
        else:
            self.DA = None
        self.conv2 = conv(n_feat, n_feat, kernel_size, bias=bias)

        self.CA = CALayer(n_feat, reduction, bias=bias)
        # self.body = nn.Sequential(*modules_body)

    def forward(self, x, index_emb=None):
        # res = self.body(x)
        res = self.act1(self.conv1(x))
        if self.DA:
            res = self.DA(x, index_emb)
        res = self.conv2(res)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, type_emb_dim=None):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]

        self.encoder_level1 = nn.ModuleList(self.encoder_level1)
        self.encoder_level2 = nn.ModuleList(self.encoder_level2)
        self.encoder_level3 = nn.ModuleList(self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None, index_emb=None):
        enc1 = x
        for layer in self.encoder_level1:
            enc1 = layer(enc1, index_emb)

        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)
        enc2 = x
        for layer in self.encoder_level2:
            enc2 = layer(enc2, index_emb)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)
        enc3 = x
        for layer in self.encoder_level3:
            enc3 = layer(enc3, index_emb)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, type_emb_dim=None):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(2)]

        self.decoder_level1 = nn.ModuleList(self.decoder_level1)
        self.decoder_level2 = nn.ModuleList(self.decoder_level2)
        self.decoder_level3 = nn.ModuleList(self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs, index_emb=None):
        enc1, enc2, enc3 = outs
        dec3 = enc3
        for layer in self.decoder_level3:
            dec3 = layer(dec3, index_emb)

        x = self.up32(dec3, self.skip_attn2(enc2, index_emb))
        dec2 = x
        for layer in self.decoder_level2:
            dec2 = layer(dec2, index_emb)

        x = self.up21(dec2, self.skip_attn1(enc1, index_emb))
        dec1 = x
        for layer in self.decoder_level1:
            dec1 = layer(dec1, index_emb)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab, type_emb_dim=None):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_dim) for _ in range(num_cab)]
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.body = nn.ModuleList(modules_body)

    def forward(self, x, index_emb=None):
        res = x 
        for layer in self.body:
            res = layer(res, index_emb)
        res = self.conv1(res)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab, type_emb_dim=None):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab, type_emb_dim=type_emb_dim)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab, type_emb_dim=type_emb_dim)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab, type_emb_dim=type_emb_dim)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs, index_emb=None):
        x = self.orb1(x, index_emb)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x, index_emb)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x, index_emb)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class MPRNet_MH(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, degrade_num=4, use_type_emb=False,
            scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(MPRNet_MH, self).__init__()

        if use_type_emb:
            type_emb_channel = n_feat
            self.type_emb_mlp = nn.Sequential(
                PositionalEncoding(n_feat),
                nn.Linear(n_feat, n_feat * 4),
                Swish(),
                nn.Linear(n_feat * 4, n_feat)
            )
        else:
            type_emb_channel = None
            self.type_emb_mlp = None

        act=nn.PReLU()
        self.head_conv1 =  conv(in_c, n_feat, kernel_size, bias=bias)
        self.head_conv2 =  conv(in_c, n_feat, kernel_size, bias=bias)
        self.head_conv3 =  conv(in_c, n_feat, kernel_size, bias=bias)
        self.shallow_feat1 = CAB(n_feat,kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_channel)
        self.shallow_feat2 = CAB(n_feat,kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_channel)
        self.shallow_feat3 = CAB(n_feat,kernel_size, reduction, bias=bias, act=act, type_emb_dim=type_emb_channel)

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, type_emb_dim=type_emb_channel, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, type_emb_dim=type_emb_channel)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, type_emb_dim=type_emb_channel, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, type_emb_dim=type_emb_channel)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab, type_emb_dim=type_emb_channel)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img, index):
        # degrade_type embedding
        if self.type_emb_mlp:
            index_emb = self.type_emb_mlp(index.float())
        else:
            index_emb = None
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(self.head_conv1(x1ltop_img), index_emb)
        x1rtop = self.shallow_feat1(self.head_conv1(x1rtop_img), index_emb)
        x1lbot = self.shallow_feat1(self.head_conv1(x1lbot_img), index_emb)
        x1rbot = self.shallow_feat1(self.head_conv1(x1rbot_img), index_emb)
        
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop, index_emb=index_emb)
        feat1_rtop = self.stage1_encoder(x1rtop, index_emb=index_emb)
        feat1_lbot = self.stage1_encoder(x1lbot, index_emb=index_emb)
        feat1_rbot = self.stage1_encoder(x1rbot, index_emb=index_emb)
        
        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]
        
        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top, index_emb)
        res1_bot = self.stage1_decoder(feat1_bot, index_emb)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2) 
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(self.head_conv2(x2top_img), index_emb)
        x2bot  = self.shallow_feat2(self.head_conv2(x2bot_img), index_emb)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top, index_emb)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot, index_emb)

        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2, index_emb)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(self.head_conv3(x3_img), index_emb)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2, index_emb)

        stage3_img = self.tail(x3_cat)

        return [stage3_img+x3_img, stage2_img, stage1_img]
