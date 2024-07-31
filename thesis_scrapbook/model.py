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

import argparse
from torch.autograd import Variable

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



class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class Self_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
        return out

class Spatial_Cross_Attention_with_Mirrored_Sim(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Spatial_Cross_Attention_with_Mirrored_Sim,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)

        self.query_conv_mirror = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv_mirror = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)

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
        proj_query  = self.query_conv(enc_feature).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(dec_feature).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # energy = self.att_gamma * energy
        attention = self.tanh(energy) # BX (N) X (N) 
        # print(f"this is spatial cross attention map shape {attention.shape}")

        proj_query_mirror  = self.query_conv(enc_feature).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key_mirror =  self.key_conv(torch.flip(enc_feature, [-1])).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy_mirror =  torch.bmm(proj_query_mirror,proj_key_mirror) # transpose check
        # energy = self.att_gamma * energy
        attention_mirror = self.tanh(energy_mirror) # BX (N) X (N) 
        # print(f"this is spatial cross attention map shape {attention.shape}")

        attention = attention * attention_mirror

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
        return out

class Channel_Cross_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Channel_Cross_Attention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))
        # self.att_gamma = nn.Parameter(torch.zeros(1))
        self.se_enc = SELayer(in_dim)
        self.se_dec = SELayer(in_dim)

        # self.softmax  = nn.Softmax(dim=1) #
        self.tanh = nn.Tanh()
    def forward(self,enc_feature, dec_feature):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = enc_feature.size()
        enc_feature = self.se_enc(enc_feature)
        dec_feature = self.se_dec(dec_feature)
        proj_query  = self.query_conv(enc_feature).view(m_batchsize,-1,width*height) # B X CX(N)
        proj_key =  self.key_conv(dec_feature).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # energy = self.att_gamma * energy
        attention = self.tanh(energy) # BX (N) X (N) 
        # print(f"this is channel attention map shape {attention.shape}")
        proj_value = self.value_conv(enc_feature).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(attention,proj_value )
        # print(f"this is the value shape {proj_value.shape}")
        out = out.view(m_batchsize,C,width,height)
        
        # out = torch.cat((self.gamma*out, dec_feature), dim=1)
        # out = self.gamma*out + dec_feature
        out = self.gamma*out
        return out

class Channel_Spatial_Cross_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Channel_Spatial_Cross_Attention,self).__init__()
        self.spatial_cross = Spatial_Cross_Attention(in_dim)
        self.channel_cross = Channel_Cross_Attention(in_dim)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,enc_feature, dec_feature):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        spatial_cross_features = self.spatial_cross(enc_feature, dec_feature)
        channel_cross_features = self.channel_cross(enc_feature, dec_feature)
        out = spatial_cross_features + channel_cross_features + (self.gamma * dec_feature)
        return out


class CNN_layer(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=3, padding=1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_layer_gnorm(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_layer_gnorm, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=3, padding=1),
                nn.GroupNorm(get_groups(nout), nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_layer_with_coor(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_layer_with_coor, self).__init__()
        self.main = nn.Sequential(
                CoordConv2d(nin, nout, kernel_size=3, padding=1, with_r=True),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_layer_with_coor_gnorm(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_layer_with_coor_gnorm, self).__init__()
        self.main = nn.Sequential(
                CoordConv2d(nin, nout, kernel_size=3, padding=1, with_r=True),
                nn.GroupNorm(get_groups(nout), nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_Encoder(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_Encoder, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.MaxPool2d(2)

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d)

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d)

        h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m_att)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d)

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d)

        h5_m_att = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m_att.view(h5_m_att.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar

class CNN_EncoderNoAtt(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_EncoderNoAtt, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.MaxPool2d(2)

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d)

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d)

        # h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d)

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d)

        # h5_m_att = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m.view(h5_m.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar

class CNN_EncoderWithCoor(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_EncoderWithCoor, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer_with_coor(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.MaxPool2d(2)

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)  # 256 x 256 16

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d) # 128 x 128 32

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d) # 64 x 64 64

        h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m_att)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d) # 32 x 32 128

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d) # 16 x 16 256

        h5_m_att = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m_att.view(h5_m_att.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar

class CNN_EncoderWithCoorAvgPool(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_EncoderWithCoorAvgPool, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer_with_coor(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.AvgPool2d((2,2))

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)  # 256 x 256 16

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d) # 128 x 128 32

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d) # 64 x 64 64

        h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m_att)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d) # 32 x 32 128

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d) # 16 x 16 256

        h5_m_att = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m_att.view(h5_m_att.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar

class CNN_EncoderWithCoorAvgPool_disent(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_EncoderWithCoorAvgPool_disent, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer_with_coor(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.AvgPool2d((2,2))

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)  # 256 x 256 16

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d) # 128 x 128 32

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d) # 64 x 64 64

        h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m_att)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d) # 32 x 32 128

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d) # 16 x 16 256 before

        h5_m_att_ori = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m_att_ori.view(h5_m_att_ori.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar, h5_m_att_ori


class CNN_EncoderWithCoorAvgPool_disent_gnorm(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(CNN_EncoderWithCoorAvgPool_disent_gnorm, self).__init__()
        self.attention_1 = Self_Attention( 64)
        self.attention_2 = Self_Attention( 256)
        self.layer1 = nn.Sequential(
            CNN_layer_with_coor_gnorm(1, 16))
        self.layer2 = nn.Sequential(
            CNN_layer_gnorm(16, 32))
        self.layer3 = nn.Sequential(
            CNN_layer_gnorm(32, 64))
        self.layer4 = nn.Sequential(
            CNN_layer_gnorm(64, 128))
        self.layer5 = nn.Sequential(
            CNN_layer_gnorm(128, 256))
        self.mu_net = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_net = nn.Linear(256 * 8 * 8, latent_dim)    
        self.DO2d = nn.Dropout2d(0.25)
        self.MP2d = nn.AvgPool2d((2,2))

    
    def reparameterize(self, mu, logvar):
        # raise NotImplementedError

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        # x shape = [N x T x C x H x W]
        out = x
        h1 = self.layer1(out)
        h1_d = self.DO2d(h1)
        h1_m = self.MP2d(h1_d)  # 256 x 256 16

        h2 = self.layer2(h1_m)
        h2_d = self.DO2d(h2)
        h2_m = self.MP2d(h2_d) # 128 x 128 32

        h3 = self.layer3(h2_m)
        h3_d = self.DO2d(h3)
        h3_m = self.MP2d(h3_d) # 64 x 64 64

        h3_m_att = self.attention_1(h3_m)

        h4 = self.layer4(h3_m_att)
        h4_d = self.DO2d(h4)
        h4_m = self.MP2d(h4_d) # 32 x 32 128

        h5 = self.layer5(h4_m)
        h5_d = self.DO2d(h5)
        h5_m = self.MP2d(h5_d) # 16 x 16 256 before

        h5_m_att_ori = self.attention_2(h5_m)

        # print('out shape after layer 5', h5_m.shape)
        h5_m_att = h5_m_att_ori.view(h5_m_att_ori.size(0), -1)
        out_mu = self.mu_net(h5_m_att)
        out_logvar = self.logvar_net(h5_m_att)
        z = self.reparameterize(out_mu, out_logvar)
        return z.view(x.shape[0], int(z.shape[0]/x.shape[0]), z.shape[1]), [h1, h2, h3, h4, h5], out_mu, out_logvar, h5_m_att_ori

class CNN_D_layer(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_D_layer, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_D_layer_gnorm(nn.Module):
    def __init__(self, nin, nout):
        super(CNN_D_layer_gnorm, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(get_groups(nout), nout),
                nn.LeakyReLU()
                )

    def forward(self, input):
        return self.main(input)

class CNN_Decoder_Skip3Only(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_Skip3Only, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        # d3 = self.upc3(d2_att) # 32 x 32
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])

        d5 = self.upc5(torch.add(d4,zero_d4)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_Decoder_SkipFront3(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SkipFront3, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        # d3 = self.upc3(d2_att) # 32 x 32
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")W
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])

        d5 = self.upc5(torch.add(d4,zero_d4)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.add(d5,zero_d5))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(torch.add(d6,skip[0])) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_Decoder_SkipFront2(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SkipFront2, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32*2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16*2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        # d3 = self.upc3(d2_att) # 32 x 32
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")W
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])

        d5 = self.upc5(torch.cat([zero_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_Decoder_SpatialCrossSkip3Only(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3Only, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        # cross_d4, enc_feat, dec_feat = self.cross_32(zero_d4, d4)
        cross_d4 = self.cross_32(zero_d4, d4)

        d5 = self.upc5(cross_d4) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        # return output, enc_feat, dec_feat
        return output

class CNN_Decoder_SpatialCrossSkipWithMirroredSim3Only(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkipWithMirroredSim3Only, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention_with_Mirrored_Sim(32)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)

        d5 = self.upc5(cross_d4) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLoss(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLoss, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossNoSigmoid(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossNoSigmoid, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                # nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256 * 2, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        # self.zero_8 = nn.Conv2d(256, 128, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _, disent_z = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")

        # zero_d1 = self.zero_8(disent_z)

        d2 = self.upc2(torch.cat([disent_z, d1], 1)) # 8 -> 16 128
        # d2 = self.upc2(d1)
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")

        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent_nosigmoid(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent_nosigmoid, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256 * 2, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1)
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        # self.zero_8 = nn.Conv2d(256, 128, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _, disent_z = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")

        # zero_d1 = self.zero_8(disent_z)

        d2 = self.upc2(torch.cat([disent_z, d1], 1)) # 8 -> 16 128
        # d2 = self.upc2(d1)
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")

        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent_gnorm(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent_gnorm, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.GroupNorm(get_groups(256), 256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer_gnorm(256 * 2, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer_gnorm(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer_gnorm(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer_gnorm(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer_gnorm(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        # self.zero_8 = nn.Conv2d(256, 128, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _, disent_z = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")

        # zero_d1 = self.zero_8(disent_z)

        d2 = self.upc2(torch.cat([disent_z, d1], 1)) # 8 -> 16 128
        # d2 = self.upc2(d1)
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")

        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossConcatenate3OnlyMiddleLossWith128Skip(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossConcatenate3OnlyMiddleLossWith128Skip, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        # cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([zero_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128256Skip(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128256Skip, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32 * 2, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16 * 2, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16 * 2, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Spatial_Cross_Attention(32)
        self.conv_32 =  nn.Sequential(
            CNN_layer(64, 64))
        self.conv_out_32 =  nn.Sequential(
            CNN_layer(64, 1))
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3)
        
        # add out branch here for premature loss calculation
        x_32 = self.conv_32(d3_att)
        x_out_32 = self.conv_out_32(x_32)
        # print(f"this is the x_64 shape {x_64.shape}")
        # print(f"this is the x_out_64 shape {x_out_64.shape}")

        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(x_32) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)
        # print(f"this is the cross skip out shape {cross_d4.shape}")

        d5 = self.upc5(torch.cat([cross_d4, d4], 1)) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(torch.cat([zero_d5, d5], 1))# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(torch.cat([skip[0], d6], 1)) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output, x_out_32

class CNN_Decoder_ChannelCrossSkip3Only(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_ChannelCrossSkip3Only, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Channel_Cross_Attention(32)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)

        d5 = self.upc5(cross_d4) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_Decoder_ChannelSpatialCrossSkip3Only(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_Decoder_ChannelSpatialCrossSkip3Only, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))

        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

        # 1x1 conv to adjust for skip
        self.zero_128 = nn.Conv2d(32, 16, 1)
        self.zero_64 = nn.Conv2d(64, 32, 1)
        self.cross_32 = Channel_Spatial_Cross_Attention(32)
    def forward(self, input):
        vec, skip, _, _ = input
        # for i, x in enumerate(skip):
        #     print(f"this is {i} skip size {x.shape}")
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8 256
        # print(f"this is d1 skip size {d1.shape}")
        d2 = self.upc2(d1) # 8 -> 16 128
        d2_att = self.attention_1(d2)
        # print(f"this is d2_att skip size {d2_att.shape}")
        d3 = self.upc3(d2_att) # 32 x 32 64
        d3_att = self.attention_2(d3) 
        # print(f"this is d3_att skip size {d3_att.shape}")
        d4 = self.upc4(d3_att) # 64 x 64 32
        # print(f"this is d4 skip size {d4.shape}")

        zero_d4 = self.zero_64(skip[2])
        cross_d4 = self.cross_32(zero_d4, d4)

        d5 = self.upc5(cross_d4) # 128 x 128 16
        # print(f"this is d5 skip size {d5.shape}")

        # zero_d5 = self.zero_128(skip[1])

        d6 = self.upc6(d5)# 256 x 256 16
        # print(f"this is d6 skip size {d6.shape}")

        output = self.upc7(d6) # 256 x 256 1
        # print(f"this is output skip size {output.shape}")
        return output

class CNN_DecoderNoSkip(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_DecoderNoSkip, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        # self.upc3 = nn.Sequential(CNN_D_layer(128 * 3, 64))
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))
        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

    def forward(self, input):
        vec, skip, _, _ = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8
        d2 = self.upc2(d1) # 8 -> 16
        d2_att = self.attention_1(d2)
        d3 = self.upc3(d2_att) # 32 x 32
        # d3 = self.upc3(torch.cat([d2_att, skip[4]], 1))
        d3_att = self.attention_2(d3)
        d4 = self.upc4(d3_att) # 64 x 64
        d5 = self.upc5(d4) # 128 x 128
        d6 = self.upc6(d5)#256 x 256
        output = self.upc7(d6) # 64 x 64
        return output

class CNN_DecoderNoSkipNoAtt(nn.Module):
    def __init__(self, n_segment = 30, dim = 1024):
        super(CNN_DecoderNoSkipNoAtt, self).__init__()
        self.dim = dim
        self.n_segment = n_segment
        self.attention_1 = Self_Attention( 128)
        self.attention_2 = Self_Attention( 64)
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 8, 1, 0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc2 = nn.Sequential(CNN_D_layer(256, 128))

        # for the skip, change to CNN_D_layer(128, 64) to undo
        # self.upc3 = nn.Sequential(CNN_D_layer(128 * 3, 64))
        self.upc3 = nn.Sequential(CNN_D_layer(128, 64))
        self.upc4 = nn.Sequential(CNN_D_layer(64, 32))
        self.upc5 = nn.Sequential(CNN_D_layer(32, 16))
        self.upc6 = nn.Sequential(CNN_D_layer(16, 16))
        self.upc7 = nn.Sequential(
                CNN_layer(16, 16),
                nn.ConvTranspose2d(16, 1, 3, 1, 1),
                # was Sigmoid in the attempt 1 to 3
                nn.Sigmoid()
                )

    def forward(self, input):
        vec, skip, _, _ = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 8
        d2 = self.upc2(d1) # 8 -> 16
        d3 = self.upc3(d2) # 32 x 32
        d4 = self.upc4(d3) # 64 x 64
        d5 = self.upc5(d4) # 128 x 128
        d6 = self.upc6(d5)#256 x 256
        output = self.upc7(d6) # 64 x 64
        return output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

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
    encoder = CNN_EncoderWithCoorAvgPool_disent_gnorm().cuda()
    decoder = CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip_disent_gnorm().cuda()
    siam_enc = TripletNet(encoder)
    siam_dec = TripletNet(decoder)
    # z = torch.randn(64, 32)
    c = torch.randn(4, 1, 256, 256).cuda()
    d = torch.randn(4, 1, 256, 256).cuda()
    e = torch.randn(4, 1, 256, 256).cuda()
    z, skip, mu, logvar, disent_z = encoder(c)
    print("encoder last output ", z.shape)
    out_decoder = decoder([z, skip, mu, logvar, disent_z])
    print(out_decoder[0].shape)
    enc1, enc2, enc3 = siam_enc(c,d,e)
    print(f"this is enc1 z {enc1[0].shape} and this is enc2 z {enc2[0].shape}")
    out1, out2, out3 = siam_dec(enc1, enc2, enc3)
    print(f"this is dec1 out {out1[0].shape} and this is dec1 size 64 out {out1[1].shape}")

    enc_total_params = sum(p.numel() for p in siam_enc.parameters())
    dec_total_params = sum(p.numel() for p in siam_dec.parameters())
    print(enc_total_params + dec_total_params )