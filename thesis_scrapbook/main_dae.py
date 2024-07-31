from ast import Sub
from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import argparse
import numpy as np
import scipy.io
import requests
import math
import os
from PIL import *
from os import listdir
from glob import glob
from os.path import isfile, isdir, join
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.model_selection import StratifiedGroupKFold
torch.cuda.empty_cache()
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataset import SkullStrippedinit_datasets, init_dataset_loader, init_dataset_loader_nocycle, AnomalousMRIDataset
from model_dae import UNet
from model_dae_enhance import DAE_ori_unroll
from utils import *
import pytorch_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *

def noise(x, args):

        # init noise parameters
        noise_res = args.noise_res
        noise_std = args.noise_std

        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[256, 256])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(256))
        roll_y = random.choice(range(256))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns

        return res

def loss_f(batch, batch_results):

        y = batch
        mask = batch.sum(dim=1, keepdim=True) > 0.01

        return (torch.pow(batch_results - y, 2) * mask.float()).mean()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name" , type = str , default = 'CNN_model')
    parser.add_argument("-model_pretrain" , default =False)
    # parser.add_argument("-train_data" , type = str , default= '/home/bspubuntu/Desktop/Research_base/data/CT_pytorch_dataset')
    # parser.add_argument("-valid_data" , type = str , default='/home/bspubuntu/Desktop/Research_base/data/CT_pytorch_testing_dataset')
    # parser.add_argument("-trained_model" , type = str , default='vol_checkpoint_CNN_model_20220209_1613 best.pt')
    parser.add_argument('--model_dir', default='weight', help='base directory to save w')
    parser.add_argument('--mod_model_dir', default='mod_weight', help='base directory to save w')
    parser.add_argument("-epoch" , type = int , default = 3000)
    parser.add_argument("-save_weight_interval" , type = int , default = 100)
    parser.add_argument("-batch_size" , type = int , default = 24)
    parser.add_argument("-Learning_rate" , type = float , default = 1e-4)
    parser.add_argument("-lr_gamma" , type = float , default = 0.977)
    parser.add_argument("-momentum" , type = float , default = 0.9)
    parser.add_argument("-weight_decay" , type = float , default = 5e-9)
    parser.add_argument("-class_threshold" , type = float , default = 0.5)
    parser.add_argument("-mode" , type = str , default = 'train')
    parser.add_argument("-thresh" , type = float , default = 0.1)
    parser.add_argument("-sample_count" , type = int , default = 8)
    
    parser.add_argument("-noise_res" , type = int , default = 16)
    parser.add_argument("-noise_std" , type = int , default = 0.2)
    parser.add_argument("-seed" , type = int , default = 0)

    args = parser.parse_args()
    sample_count = args.sample_count
    if(args.sample_count > args.batch_size):
        sample_count = args.batch_size
    

    #Create output folder
    mkdir(args.model_dir)
    mkdir(args.mod_model_dir)

    # dd/mm/YY H:M:S
    # datetime object containing current date and time
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.timestamp=dt_string
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print ("-------------------------------------")
        print ("GPU is ready")

    training_dataset, testing_dataset = SkullStrippedinit_datasets()
    # training_dataset_loader = init_dataset_loader(training_dataset, args.batch_size)
    # testing_dataset_loader = init_dataset_loader(testing_dataset, args.batch_size)
    training_dataset_loader = torch.utils.data.DataLoader(
                    training_dataset,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True
                    )
    testing_dataset_loader = torch.utils.data.DataLoader(
                    testing_dataset,
                    batch_size= 8, shuffle=False,
                    num_workers=0, drop_last=True
                    )
    
    # evaluation dataset
    DATASET_PATH = '../23_01_15_AnoDDPM/AnoDDPM/DATASETS/CancerousDataset/BraTS_T2'
    a_set = AnomalousMRIDataset(DATASET_PATH)
    eval_dataset_loader = torch.utils.data.DataLoader(
                    a_set,
                    batch_size= 1, shuffle=False,
                    num_workers=0, drop_last=True
                    )

    # Announce the model
    start_epoch = 0
    Unet = DAE_ori_unroll()
    Unet = nn.DataParallel(Unet, device_ids=[0,1,3])

    # Unet.load_state_dict(torch.load("./mod_weight/Unet110.pt"))
    # start_epoch = 110
    Unet.to(device)

    optimiser = torch.optim.Adam(Unet.parameters(), lr=args.Learning_rate, amsgrad=True, weight_decay=0.00001)
    lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=100)

    lr_scheduler.step(start_epoch)

    SSIM_loss = pytorch_ssim.SSIM()
    SSIM_loss.cuda()

    t_log_dir = "./Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))
    tfr = 0
    progress = tqdm(total=args.epoch)
    for epoch in range(start_epoch, start_epoch + args.epoch + 1):
        Unet.train()
        epoch_Unet_loss = []
        
        batch = tqdm(training_dataset_loader,ncols=80)
        for idx, x in enumerate(batch):
            # x = x.float().cuda()
            for key, val in x.items():
                x[key] = val.to(device)

            optimiser.zero_grad()

            # for original noise
            noised_batch = add_coarse_noise(x["anchor"].clone())
            batch_result = Unet(noised_batch)

            # loss = loss_f(x["anchor"], batch_result)

            # for fair noise uses
            # noised_batch = x["noisy_anchor"]
            # batch_result = Unet(noised_batch)

            # loss = loss_f(x["anchor"], batch_result)

            # last discussion comparison unet with SSIM see how much it can help
            SSIM_neg = SSIM_loss(batch_result, x["anchor"])
            loss = loss_f(x["anchor"], batch_result) - (SSIM_neg)

            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(Unet.parameters(), max_norm=0.5)
            optimiser.step()

            epoch_Unet_loss.append(loss.detach().cpu().numpy())

            # save the training recontructed images for every 5th epoch and index == 0
            if(int(epoch) % args.save_weight_interval == 0 and idx == 0):
                print("saving training images samples")
                create_folder_save_tensor_imgs("train_data_out/anchor", sample_count, f"train_data_epoch{epoch}.png", x["anchor"][-sample_count:, ...] , batch_result[-sample_count:, ...])
                create_folder_save_tensor_imgs("train_data_out/negative", sample_count, f"train_data_epoch{epoch}.png", noised_batch[-sample_count:, ...] , batch_result[-sample_count:, ...])

        log_writer.add_scalar('epoch_Unet_loss', np.mean(np.array(epoch_Unet_loss)), epoch)

        print(f"epoch {epoch} \n")
        print("epoch Unet MSE loss: ", np.mean(np.array(epoch_Unet_loss)))

        progress.update(1)
        lr_scheduler.step()
        if(int(epoch) % args.save_weight_interval == 0):
            # run eval and save evaluation image here
            with torch.no_grad():
                print("saving evaluation images samples")
                for x in testing_dataset_loader:
                    for key, val in x.items():
                        x[key] = val.to(device)
                
                noised_batch = add_coarse_noise(x["anchor"].clone())
                batch_result = Unet(noised_batch)
                # noised_batch = x["noisy_anchor"]
                # batch_result = Unet(noised_batch)

                create_folder_save_tensor_imgs("eval_data_out/anchor", sample_count, f"eval_data_epoch{epoch}.png", x["anchor"][-sample_count:, ...] , batch_result[-sample_count:, ...])
                create_folder_save_tensor_imgs("eval_data_out/negative", sample_count, f"eval_data_out{epoch}.png", noised_batch[-sample_count:, ...] , batch_result[-sample_count:, ...])
            if(int(epoch) >= 900 or int(epoch) == 0):
                print("saving model weights .... ")
                weight_path = './weight/Unet%d.pt' %(int(epoch))
                torch.save(Unet.module.state_dict(), weight_path)

                weight_path = './mod_weight/Unet%d.pt' %(int(epoch))
                torch.save(Unet.state_dict(), weight_path)

                torch.cuda.empty_cache()