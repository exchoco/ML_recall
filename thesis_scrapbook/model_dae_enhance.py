import torch
import os
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.modules.conv as conv
from math import sqrt
import torch.nn.functional as F

import argparse
from torch.autograd import Variable


class Spatial_Cross_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Spatial_Cross_Attention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        # self.att_gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1) #
        self.tanh = nn.Tanh()
        # self.conv_final = nn.Conv2d(in_channels = in_dim*2, out_channels = in_dim , kernel_size= 1)
        # self.se_final = SELayer(in_dim*2)
    def forward(self,enc_feature, dec_feature):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = enc_feature.size()
        proj_query_keepshape  = self.query_conv(enc_feature)
        proj_query = proj_query_keepshape.view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key_keepshape =  self.key_conv(dec_feature)
        proj_key = proj_key_keepshape.view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # energy = self.att_gamma * energy
        attention = self.tanh(energy) # BX (N) X (N) 
        # print(f"this is spatial cross attention map shape {attention.shape}")
        proj_value = self.value_conv(enc_feature).view(m_batchsize,-1,width*height) # B X C X N
        # print(f"proj value shape {proj_value.shape}")
        out = torch.bmm(attention,proj_value.permute(0,2,1))
        # print(f"this is the out shape {out.shape}")
        out = out.permute(0,2,1).view(m_batchsize,C,width,height)
        
        # out = torch.cat((out, dec_feature), dim=1)
        # out = self.se_final(out)
        # out = self.conv_final(out)
        out = self.gamma*out + self.gamma2*dec_feature
        # out = self.gamma*out
        # return out, proj_query_keepshape, proj_key_keepshape
        # return out, attention
        return out

class WNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)

def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]

class UNetUpBlock_noSkip(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock_noSkip, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size//2, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, in_x):
        x, bridge = in_x
        up = self.up(x)
        out = self.conv_block(up)

        return out

class UNetUpBlock_noSkip_latent(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock_noSkip_latent, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, in_size//2, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size//2, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, in_x):
        x, bridge = in_x
        up = self.up(x)
        out = self.conv_block(up)

        return out

class UNetUpBlock_withSkip(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock_withSkip, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, in_x):
        x, bridge = in_x
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

class UNetUpBlock_CrossSkip(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock_CrossSkip, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)
        self.cross_32 = Spatial_Cross_Attention(out_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, in_x):
        x, bridge = in_x
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        # print(f"cross att shape pair {up.shape} {crop1.shape}")
        # out, attn_map = self.cross_32(crop1, up)
        out = self.cross_32(crop1, up)
        cross_out_backup = out
        out = torch.cat([up, out], 1)
        # print(f"cross out shape {out.shape}")
        out = self.conv_block(out)
        # return out
        return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class VAE_withCrossSkip128skipMiddleLossDisent(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossDisent, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)
        self.last_enc_channel = prev_channels
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            if(i>=3 or i ==0 ):
                self.up_path.append(
                    UNetUpBlock_noSkip(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
                )
            elif(i==2):
                self.up_path.append(
                    UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
                )
            elif(i==1):
                self.up_path.append(
                    UNetUpBlock_withSkip(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
                )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            # print(f"{i}: {x.shape}")
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)

        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, x, blocks):

        # recover the z shape
        x = self.latent_up(x)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            if(i==2 or i ==3):
                x = up(x, skip)
            else:
                # print(x.shape, skip.shape)
                x = up(x, skip)
            # print(f"{i}: {x.shape} and {skip.shape}")

        return x

    def forward_without_last(self, x):
        x, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        z_backup = x
        x = self.forward_up_without_last(x, blocks)
        return x, z_backup, out_mu, out_logvar, z_ori

    def forward(self, x):
        x, z_backup, out_mu, out_logvar, z_ori = self.get_features(x)
        return self.last(x), z_backup, out_mu, out_logvar

    def get_features(self, x):
        return self.forward_without_last(x)

class DAE_ori_unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(DAE_ori_unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        return x, blocks

    def forward_up_without_last(self, x, blocks):

        # recover the z shape
        # x = self.latent_up(z)
        # x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))

        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5

    def forward(self, x):
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks)
        return self.last(x)

class DAE_ori_unroll_forshow(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(DAE_ori_unroll_forshow, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        return x, blocks

    def forward_up_without_last(self, x, blocks):

        # recover the z shape
        # x = self.latent_up(z)
        # x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))

        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        forshow = d_3

        return d_5, forshow

    def forward(self, x):
        x, blocks = self.forward_down(x)
        x, forshow = self.forward_up_without_last(x, blocks)
        return self.last(x), forshow

class VAE_withCrossSkip128skipMiddleLossDisent_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossDisent_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_out_32 = nn.Sequential(UNetConvBlock(prev_channels, 1, padding, norm=norm))

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))

        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_out_32(d_2)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32, z_ori


class VAE_withCrossSkip128skipMiddleLoss_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLoss_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_out_32 = nn.Sequential(UNetConvBlock(prev_channels, 1, padding, norm=norm))

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_out_32(d_2)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32


class VAE_withCrossSkip128skipMiddleLossConv_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32


class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x)
        # print(f"after layer one shape, {layer1_out.shape}")
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        # print(f"after maxpool one shape, {layer1_out_m.shape}")
        
        layer2_out = self.layer2(layer1_out_m)
        # print(f"after layer two shape, {layer2_out.shape}")
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)
        # print(f"after maxpool two shape, {layer2_out_m.shape}")

        layer3_out = self.layer3(layer2_out_m)
        # print(f"after layer three shape, {layer3_out.shape}")
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)
        # print(f"after maxpool three shape, {layer3_out_m.shape}")

        layer4_out = self.layer4(layer3_out_m)
        # print(f"after layer four shape, {layer4_out.shape}")
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)
        # print(f"after maxpool four shape, {layer4_out_m.shape}")

        layer5_out = self.layer5(layer4_out_m)
        # print(f"after layer five shape, {layer5_out.shape}")
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)
        # print(f"after maxpool five shape, {layer5_out_m.shape}")

        layer6_out = self.layer6(layer5_out_m)
        # print(f"after layer six shape, {layer6_out.shape}")
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # print(f"after up layer one shape, {d_1.shape}")
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        # print(f"layer two skip shape, {skip.shape}")
        d_2 = self.up2((d_1, skip))
        # print(f"after up layer two shape, {d_2.shape}")

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        # print(f"layer three skip shape, {skip.shape}")
        d_3 = self.up3((d_2, skip))
        # print(f"after up layer three shape, {d_3.shape}")

        skip = blocks[-5]
        # print(f"layer four skip shape, {skip.shape}")
        d_4 = self.up4((d_3, skip))
        # print(f"after up layer four shape, {d_4.shape}")

        skip = blocks[-6]
        # print(f"layer five skip shape, {skip.shape}")
        d_5 = self.up5((d_4, skip))
        # print(f"after up layer five shape, {d_5.shape}")

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss_forshow(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss_forshow, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        forshow = d_3

        return d_5, x_out_32, forshow

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32, forshow = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32, forshow

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((z_ori, skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        # decoder upblock



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        # last_x = x.view(x.size(0), -1)
        # out_mu = self.mu_net(last_x)
        # out_logvar = self.logvar_net(last_x)
        # z = self.reparameterize(out_mu, out_logvar)
        return None, blocks, None, None, x

    def forward_up_without_last(self, z, blocks, z_ori):

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((z_ori, skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32
    
class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS_forshow(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS_forshow, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        # decoder upblock



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        # last_x = x.view(x.size(0), -1)
        # out_mu = self.mu_net(last_x)
        # out_logvar = self.logvar_net(last_x)
        # z = self.reparameterize(out_mu, out_logvar)
        return None, blocks, None, None, x

    def forward_up_without_last(self, z, blocks, z_ori):

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((z_ori, skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        forshow = d_3

        return d_5, x_out_32, forshow

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32, forshow = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32, forshow

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet_forshow(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet_forshow, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((z_ori, skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        forshow = d_3

        return d_5, x_out_32, forshow

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32, forshow = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32, forshow

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_forshow(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_forshow, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3, uped_generated = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        # forshow = d_3
        forshow = d_2

        return d_5, x_out_32, forshow

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32, forshow = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32, forshow

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_maxpool(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_maxpool, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.max_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.max_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.max_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.max_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.max_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_concat(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_concat, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_noskip(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_noskip, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels * 2, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_vvae(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_vvae, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x)
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m)
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m)
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m)
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        # d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Conv2d(self.last_enc_channel, self.last_enc_channel, kernel_size=3)
        self.logvar_net = nn.Conv2d(self.last_enc_channel, self.last_enc_channel, kernel_size=3)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        # last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(x)
        out_logvar = self.logvar_net(x)
        z = self.reparameterize(out_mu, out_logvar)
        print(f"this is reparameterized z shape {z.shape}")
        return z, blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        # x = self.latent_up(z)
        # x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        # d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        d_1 = self.up1((z, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial_large(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=4,
            wf=6,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial_large, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        # self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        # prev_channels = 2 ** (wf + 4)
        
        # self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        # prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        # self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        # prev_channels = 2 ** (wf + 4)

        # self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        # prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = WNConv2d(self.last_enc_channel, self.last_enc_channel, kernel_size=3)
        self.logvar_net = WNConv2d(self.last_enc_channel, self.last_enc_channel, kernel_size=3)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        print(f"this is before mean shape {layer4_out.shape}")


        x = layer4_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        # last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(x)
        print(f"this is mean shape {out_mu.shape}")
        out_logvar = self.logvar_net(x)
        z = self.reparameterize(out_mu, out_logvar)
        print(f"this is reparameterized z shape {z.shape}")
        return z, blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        # x = self.latent_up(z)
        # x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        # skip = blocks[-2]
        # # print(skip.shape)
        # # print(x.shape)
        # # d_1 = self.up1((torch.cat([x, z_ori], 1), skip))
        # d_1 = self.up1((z, skip))


        # skip = blocks[-3]
        # d_2 = self.up2((d_1, skip))

        # x_out_32 = self.conv_32(d_2)
        # x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-2]
        d_3 = self.up3((z, skip))

        skip = blocks[-3]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-4]
        d_5 = self.up5((d_4, skip))

        return d_5, d_5

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32


class VAE_withCrossSkip128skipMiddleLossConv_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkip128skipMiddleLossConv_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_withSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32

class VAE_withCrossSkipMiddleLossConv_Unroll(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=6,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            latent_dim=1024):

        super(VAE_withCrossSkipMiddleLossConv_Unroll, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf), padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.layer2 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 1), padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.layer3 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 2), padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.layer4 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 3), padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.layer5 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 4), padding, norm=norm))
        prev_channels = 2 ** (wf + 4)
        
        self.layer6 = nn.Sequential(UNetConvBlock(prev_channels, 2 ** (wf + 5), padding, norm=norm))
        prev_channels = 2 ** (wf + 5)

        self.last_enc_channel = prev_channels

        self.up1 = nn.Sequential(UNetUpBlock_noSkip_latent(prev_channels, 2 ** (wf + 4), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 4)

        self.up2 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 3), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 3)

        self.conv_32 = nn.Sequential(UNetConvBlock(prev_channels, prev_channels, padding, norm=norm))
        self.conv_out_32 = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.up3 = nn.Sequential(UNetUpBlock_CrossSkip(prev_channels, 2 ** (wf + 2), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 2)

        self.up4 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf + 1), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf + 1)

        self.up5 = nn.Sequential(UNetUpBlock_noSkip(prev_channels, 2 ** (wf), up_mode, padding, norm=norm))
        prev_channels = 2 ** (wf)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # custom reparameterization
        self.last_enc_spatial = int(256 / (2**(self.depth -1)))
        # print(self.last_enc_spatial, self.last_enc_channel)
        self.mu_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)
        self.logvar_net = nn.Linear(self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial, self.latent_dim)

        # decoder upblock
        self.latent_up = nn.Linear(self.latent_dim, self.last_enc_channel * self.last_enc_spatial * self.last_enc_spatial)



    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward_down(self, x):

        blocks = []

        layer1_out = self.layer1(x) 
        blocks.append(layer1_out)
        layer1_out_m = F.avg_pool2d(layer1_out, 2)
        
        layer2_out = self.layer2(layer1_out_m) 
        blocks.append(layer2_out)
        layer2_out_m = F.avg_pool2d(layer2_out, 2)

        layer3_out = self.layer3(layer2_out_m) 
        blocks.append(layer3_out)
        layer3_out_m = F.avg_pool2d(layer3_out, 2)

        layer4_out = self.layer4(layer3_out_m) 
        blocks.append(layer4_out)
        layer4_out_m = F.avg_pool2d(layer4_out, 2)

        layer5_out = self.layer5(layer4_out_m) 
        blocks.append(layer5_out)
        layer5_out_m = F.avg_pool2d(layer5_out, 2)

        layer6_out = self.layer6(layer5_out_m)
        blocks.append(layer6_out)

        x = layer6_out
        
        # reparameterize
        # print(f"this is x shape {x.shape}")
        last_x = x.view(x.size(0), -1)
        out_mu = self.mu_net(last_x)
        out_logvar = self.logvar_net(last_x)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), blocks, out_mu, out_logvar, x

    def forward_up_without_last(self, z, blocks, z_ori):

        # recover the z shape
        x = self.latent_up(z)
        x = x.view(-1, self.last_enc_channel, self.last_enc_spatial, self.last_enc_spatial)
        # print(f"this is recovered x shape {x.shape}")

        skip = blocks[-2]
        # print(skip.shape)
        # print(x.shape)
        d_1 = self.up1((x, skip))


        skip = blocks[-3]
        d_2 = self.up2((d_1, skip))

        x_out_32 = self.conv_32(d_2)
        x_out_32 = self.conv_out_32(x_out_32)

        skip = blocks[-4]
        d_3 = self.up3((d_2, skip))

        skip = blocks[-5]
        d_4 = self.up4((d_3, skip))

        skip = blocks[-6]
        d_5 = self.up5((d_4, skip))

        return d_5, x_out_32

    def forward(self, x):
        z, blocks, out_mu, out_logvar, z_ori = self.forward_down(x)
        x, x_out_32 = self.forward_up_without_last(z, blocks, z_ori)
        return self.last(x), z, out_mu, out_logvar, z_ori, x_out_32



class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1 = None, x2 = None, x3 = None):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3
        
    def get_embedding(self, x):
        return self.embedding_net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial_large()
    net.to(device)
    a1 = torch.randn(4, 1, 256, 256).to(device)
    a2 = torch.randn(4, 1, 256, 256).to(device)
    a3 = torch.randn(4, 1, 256, 256).to(device)
    siam_net = TripletNet(net)
    b1, b2, b3 = siam_net(a1, a2, a3)
    print(f"this is any output shape {b1[0].shape}")
    # print(b1[0].shape, b1[1].shape, b1[2].shape, b1[3].shape, b1[4].shape, b1[5].shape, b1[6].shape)
    pytorch_total_params = sum(p.numel() for p in siam_net.parameters())
    print(pytorch_total_params)