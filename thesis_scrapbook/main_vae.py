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
from model import (CNN_Encoder, 
    CNN_EncoderWithCoor, 
    SiameseNet, 
    TripletNet, 
    CNN_Decoder_SkipFront3, 
    CNN_DecoderNoSkip,
    CNN_DecoderNoSkipNoAtt,
    CNN_Decoder_Skip3Only, 
    CNN_Decoder_SpatialCrossSkip3Only,
    CNN_Decoder_ChannelCrossSkip3Only,
    CNN_Decoder_ChannelSpatialCrossSkip3Only,
    CNN_Decoder_SpatialCrossSkip3OnlyMiddleLoss,
    CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128Skip,
    CNN_Decoder_SpatialCrossSkip3OnlyMiddleLossWith128256Skip,
    CNN_EncoderWithCoorAvgPool,
    CNN_Decoder_SpatialCrossConcatenate3OnlyMiddleLossWith128Skip,
    CNN_Decoder_SkipFront2)
from utils import *
import pytorch_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR


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
    parser.add_argument("-save_weight_interval" , type = int , default = 20)
    parser.add_argument("-batch_size" , type = int , default = 28)
    parser.add_argument("-Learning_rate" , type = float , default = 1e-4)
    parser.add_argument("-lr_gamma" , type = float , default = 0.977)
    parser.add_argument("-momentum" , type = float , default = 0.9)
    parser.add_argument("-weight_decay" , type = float , default = 5e-9)
    parser.add_argument("-class_threshold" , type = float , default = 0.5)
    parser.add_argument("-mode" , type = str , default = 'train')
    parser.add_argument("-thresh" , type = float , default = 0.1)

    args = parser.parse_args()

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
                    batch_size= 8, shuffle=True,
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
    Encoder = CNN_EncoderWithCoorAvgPool()
    Decoder = CNN_Decoder_SkipFront2()
    siam_enc = nn.DataParallel(Encoder, device_ids=[0,1,3])
    siam_dec = nn.DataParallel(Decoder, device_ids=[0,1,3])

    siam_enc.load_state_dict(torch.load("./mod_weight/TriEncoder400.pt"))
    siam_dec.load_state_dict(torch.load("./mod_weight/TriDecoder400.pt"))
    start_epoch = 400
    siam_enc.to(device)
    siam_dec.to(device)

    params =  list(siam_enc.parameters()) + list(siam_dec.parameters())
    enc_dec_optimizer = torch.optim.Adam(params, lr=args.Learning_rate, weight_decay=args.weight_decay)
    # enc_dec_scheduler = lr_scheduler.StepLR(enc_dec_optimizer, 600, gamma=0.1, last_epoch=-1)
    enc_dec_scheduler = CosineAnnealingLR(optimizer=enc_dec_optimizer, T_max=100)


    enc_optimizer = torch.optim.Adam(siam_enc.parameters(), lr=args.Learning_rate, weight_decay=args.weight_decay)
    # enc_scheduler = lr_scheduler.StepLR(enc_optimizer, 600, gamma=0.1, last_epoch=-1)
    enc_scheduler = CosineAnnealingLR(optimizer=enc_optimizer, T_max=100)

    enc_dec_scheduler.step(start_epoch)
    enc_scheduler.step(start_epoch)

    mae_criterion = nn.L1Loss(reduction = "sum")
    mae_criterion.cuda()

    SSIM_loss = pytorch_ssim.SSIM()
    SSIM_loss.cuda()

    t_log_dir = "./Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))
    tfr = 0
    progress = tqdm(total=args.epoch)
    for epoch in range(start_epoch, start_epoch + args.epoch + 1):
        siam_enc.train()
        siam_dec.train()
        epoch_enc_dec_loss = []
        epoch_enc_triplet_loss = []
        epoch_mae_anchor = []
        epoch_mae_pos = []
        epoch_mae_neg_to_ori = []
        epoch_mae_neg_to_noisy = []
        epoch_kld_anchor = []
        epoch_kld_pos = []
        epoch_kld_neg = []
        epoch_TV_anchor = []
        epoch_TV_pos = []
        epoch_TV_neg = []
        epoch_SSIM_anchor = []
        epoch_SSIM_pos = []
        epoch_SSIM_neg = []
        batch = tqdm(training_dataset_loader,ncols=80)
        for idx, x in enumerate(batch):
            # x = x.float().cuda()
            for key, val in x.items():
                x[key] = val.to(device)

            enc_dec_optimizer.zero_grad()

            out_anchor = siam_enc(x["anchor"])
            rec_anchor = siam_dec(out_anchor)

            mae_anchor = mae_criterion(rec_anchor, x["anchor"])

            kld_anchor = kl_criterion(out_anchor[2], out_anchor[3], args.batch_size)
            
            SSIM_anchor = SSIM_loss(rec_anchor, x["anchor"])

            enc_dec_loss = (mae_anchor) + (kld_anchor)

            enc_dec_loss.backward()
            enc_dec_optimizer.step()

            # save the training recontructed images for every 5th epoch and index == 0
            if(int(epoch) % args.save_weight_interval == 0 and idx == 0):
                print("saving training images samples")
                create_folder_save_tensor_imgs("train_data_out/anchor", args.batch_size // 4, f"train_data_epoch{epoch}.png", x["anchor"][-args.batch_size // 4:, ...] , rec_anchor[-args.batch_size // 4:, ...])
         
            epoch_enc_dec_loss.append(enc_dec_loss.detach().cpu().numpy())
            epoch_mae_anchor.append(mae_anchor.detach().cpu().numpy())
            epoch_kld_anchor.append(kld_anchor.detach().cpu().numpy())

        log_writer.add_scalar('epoch_enc_dec_loss', np.mean(np.array(epoch_enc_dec_loss)), epoch)
        log_writer.add_scalar('epoch_mae_anchor', np.mean(np.array(epoch_mae_anchor)), epoch)
        log_writer.add_scalar('epoch_kld_anchor', np.mean(np.array(epoch_kld_anchor)), epoch)

        print(f"epoch {epoch} \n")

        print("\nepoch encoder decoder l1 and kld loss: ", np.mean(np.array(epoch_enc_dec_loss)))

        progress.update(1)
        if(int(epoch) % args.save_weight_interval == 0):
            # run eval and save evaluation image here
            with torch.no_grad():
                print("saving evaluation images samples")
                siam_enc.eval()
                siam_dec.eval()
                for x in testing_dataset_loader:
                    for key, val in x.items():
                        x[key] = val.to(device)
                
                out_anchor = siam_enc(x["anchor"])
                rec_anchor = siam_dec(out_anchor)
                create_folder_save_tensor_imgs("eval_data_out/anchor", args.batch_size // 4, f"eval_data_epoch{epoch}.png", x["anchor"][-args.batch_size // 4:, ...] , rec_anchor[-args.batch_size // 4:, ...])

            print("saving model weights .... ")
            weight_path = './weight/TriEncoder%d.pt' %(int(epoch))
            torch.save(siam_enc.module.state_dict(), weight_path)
            weight_path = './weight/TriDecoder%d.pt' %(int(epoch))
            torch.save(siam_dec.module.state_dict(), weight_path)

            weight_path = './mod_weight/TriEncoder%d.pt' %(int(epoch))
            torch.save(siam_enc.state_dict(), weight_path)
            weight_path = './mod_weight/TriDecoder%d.pt' %(int(epoch))
            torch.save(siam_dec.state_dict(), weight_path)
            torch.cuda.empty_cache()