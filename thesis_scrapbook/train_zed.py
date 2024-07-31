'''Usage
python train_zed.py \
--datapath nccu_dataset \
--epochs 100 \
--savemodel checkpoint/nccu_dataset
'''
from __future__ import print_function
import torchvision
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.utils as utils
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
import cv2
from PIL import Image
from RT_stereov4 import HRstereoNet
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
from dataloader import zedimage as lt
from dataloader import allloader as DA
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='../../../datasets/dataset_kitti',
                    help='select model')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')                                     
parser.add_argument('--model', default='RT_stereov4',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--savemodel', default='./cpts',
                    help='save model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('There are ', torch.cuda.device_count(), 'GPUs.')

if os.path.isdir(args.savemodel) is False:
    os.makedirs(args.savemodel)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img = lt.dataloader(args.datapath)
print(f'Number images of Dataset: {len(all_left_img)} + {len(all_right_img)}')
TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img, all_right_img, True), 
         batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

if args.model == 'RT_stereov4':
    model = HRstereoNet(args.maxdisp)   
else:
    print('no model')

# print('model: ', model)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    # model.load_state_dict(state_dict['state_dict'])

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        if 'module' not in k:
            k = 'module.' + k

        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

def warp(x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            vgrid = grid.cuda()
        #vgrid = grid
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)


def cal_grad2_error(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo 
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    
    
    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))
    print(weights_x.shape)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0


def train(imgL,imgR,epoch):
        model.train()

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        optimizer.zero_grad()
        
        if args.model == 'RT_stereov4':
            # reconstructed left image
            disp_left, reconstructed_left = model(imgL,imgR)
            # rec loss
            ssim_left = SSIM(reconstructed_left[:, :, :, 75:575], imgL[:, :, :, 75:575])
            ssim_loss_left = torch.mean(ssim_left)
            loss_left = torch.mean(torch.abs(reconstructed_left[:, :, :, 75:575] - imgL[:, :, :, 75:575]))
            image_loss_left = 0.85*ssim_loss_left + 0.15*loss_left
            # smooth loss
            disp_loss = cal_grad2_error(disp_left / 20, imgL, 1.0)

            # reconstructed right image
            imgL_rot = imgL.flip(2).flip(3)
            imgR_rot = imgR.flip(2).flip(3)
            disp_right, reconstructed_right = model(imgR_rot,imgL_rot)
            disp_right = disp_right.flip(2).flip(3)
            reconstructed_right = reconstructed_right.flip(2).flip(3)
            # rec loss
            ssim_right = SSIM(reconstructed_right[:, :, :, 0:500], imgR[:, :, :, 0:500])
            ssim_loss_right = torch.mean(ssim_right)
            loss_right = torch.mean(torch.abs(reconstructed_right[:, :, :, 0:500] - imgR[:, :, :, 0:500]))
            image_loss_right = 0.85*ssim_loss_right + 0.15*loss_right

            # smooth loss
            disp_loss_2 = cal_grad2_error(disp_right / 20, imgR, 1.0) 

            ### Left-Right Consistency
            right_to_left_disp = warp(disp_right, disp_left)
            disp_right_rot = disp_right.flip(2).flip(3)
            disp_left_rot = disp_left.flip(2).flip(3)
            left_to_right_disp = warp(disp_left_rot, disp_right_rot)
            left_to_right_disp = left_to_right_disp.flip(2).flip(3)
    
            lr_right_loss = torch.mean(torch.abs(right_to_left_disp[:, :, :, 75:575] - disp_left[:, :, :, 75:575]))
            lr_left_loss = torch.mean(torch.abs(left_to_right_disp[:, :, :, 0:500] - disp_right[:, :, :, 0:500]))
            lr_loss = lr_left_loss + lr_right_loss

            # all loss
            if epoch < 15:
               loss = image_loss_left + image_loss_right
            else:
               loss = image_loss_left + image_loss_right + 0.01*lr_loss + 10*(disp_loss + disp_loss_2)

        loss.backward()
        optimizer.step()

        return loss.data


def adjust_learning_rate(optimizer, epoch):
    lr = 0.0005
    if epoch > 199:
        lr = 0.00005
    print('lr:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

        start_full_time = time.time()
        epoch_pbar = tqdm(range(0, args.epochs))
        for epoch in epoch_pbar:
           print('This is %d-th epoch' %(epoch))
           total_train_loss = 0
           adjust_learning_rate(optimizer,epoch)

           ## training ##
           iter_pbar = tqdm(TrainImgLoader)
           for batch_idx, (imgL_crop, imgR_crop) in enumerate(iter_pbar):
             start_time = time.time()

             loss = train(imgL_crop,imgR_crop,epoch)
             iter_pbar.set_description('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
             total_train_loss += loss
           epoch_pbar.set_description('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

           #SAVE
           savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'_'+str(total_train_loss/len(TrainImgLoader))[7:13]+'.tar'
           torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
            }, savefilename)

        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
       
if __name__ == '__main__':
   main()
