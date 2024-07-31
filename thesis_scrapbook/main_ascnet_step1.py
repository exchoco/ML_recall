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
from model_ascnet import (Discriminator,
                Generator)
from utils import *
import pytorch_ssim
from torchvision.utils import save_image

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
    parser.add_argument("-save_weight_interval" , type = int , default = 50)
    parser.add_argument("-batch_size" , type = int , default = 16)
    parser.add_argument("-Learning_rate" , type = float , default = 5e-5)
    parser.add_argument("-lr_gamma" , type = float , default = 0.977)
    parser.add_argument("-momentum" , type = float , default = 0.9)
    parser.add_argument("-weight_decay" , type = float , default = 5e-9)
    parser.add_argument("-class_threshold" , type = float , default = 0.5)
    parser.add_argument("-mode" , type = str , default = 'train')
    parser.add_argument("-thresh" , type = float , default = 0.1)
    parser.add_argument("-sample_count" , type = int , default = 8)
    parser.add_argument("-n_critic" , type = int , default = 1)
    parser.add_argument("-n_gen" , type = int , default = 1)
    parser.add_argument("-sample_interval" , type = int , default = 50)

    args = parser.parse_args()
    sample_count = args.sample_count
    if(args.sample_count > args.batch_size):
        sample_count = args.batch_size
    

    #Create output folder
    mkdir(args.model_dir)
    mkdir(args.mod_model_dir)
    os.makedirs("./train_data_out/anchor/", exist_ok=True)

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
                    batch_size= 1, shuffle=True,
                    num_workers=0, drop_last=True
                    )

    # Announce the model
    start_epoch = 0
    gen_model = Generator()
    disc_model = Discriminator()
    gen_model = nn.DataParallel(gen_model, device_ids=[0,1,3])
    disc_model = nn.DataParallel(disc_model, device_ids=[0,1,3])

    gen_model.load_state_dict(torch.load("./asc_step1_weight/Generator3000.pt"))
    disc_model.load_state_dict(torch.load("./asc_step1_weight/Discriminator1800.pt"))
    # start_epoch = 10

    # siam_enc.load_state_dict(torch.load("./mod_weight/TriEncoder1140.pt"))
    # siam_dec.load_state_dict(torch.load("./mod_weight/TriDecoder1140.pt"))
    # start_epoch = 1160
    gen_model.to(device)
    disc_model.to(device)

    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.Learning_rate, weight_decay=1e-5)

    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.Learning_rate,  weight_decay=1e-5)

    #mse loss for generator
    mse_criterion = nn.MSELoss(reduction = "sum")
    # mse_criterion = nn.MSELoss()
    mse_criterion.cuda()

    #mae loss for discriminator
    mae_criterion = nn.L1Loss(reduction = "sum")
    # mae_criterion = nn.L1Loss()
    mae_criterion.cuda()

    t_log_dir = "./asc_step1_Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))

    padding_epoch = len(str(args.epoch))
    padding_i = len(str(len(training_dataset_loader)))
    
    batches_done = 0
    progress = tqdm(total=args.epoch)
    for epoch in range(start_epoch, start_epoch + args.epoch + 1):
        gen_model.train()
        disc_model.train()
        epoch_GLoss = []
        G_loss_mae_validity = []
        G_loss_mse_recons = []
        G_loss_disjoincy = []
        epoch_DLoss = []
        # loop thru healthy loader
        for i, (x_healthy) in enumerate(training_dataset_loader):
            img_healthy = x_healthy["anchor"]
            img_healthy = img_healthy.to(device)
            healthy_labels = torch.ones(img_healthy.shape[0], 1) * -1
            healthy_labels = healthy_labels.to(device)
            # the following loop is of size 1 for eval
            for j, (x_tumor) in enumerate(eval_dataset_loader):
                x_tumor["image"] = x_tumor["image"].to(device)
                # solve the shape since we dont need the batch dimension
                x_tumor["image"] = x_tumor["image"].view(-1, 1, x_tumor["image"].shape[-2], x_tumor["image"].shape[-2])
                # shuffle the mini batch
                rand_perm_idx = torch.randperm(x_tumor["image"].shape[0]).to(device)
                x_tumor["image"] = torch.index_select(x_tumor["image"], 0, rand_perm_idx)

                # solved randomize and batch size for the tumor loader
                subbatch_len = int(math.floor(x_tumor["image"].shape[0] / args.batch_size))
                for idx in range(int(math.floor(x_tumor["image"].shape[0] / args.batch_size))):
                    start_idx = idx * args.batch_size
                    end_idx = (idx+1) * args.batch_size
                    img_tumor = x_tumor["image"][start_idx:end_idx]

                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    for _ in range(args.n_gen):
                        gen_optimizer.zero_grad()
                        fake_wild, fake_fence, fake_combined = gen_model(img_tumor)
                        # Real images
                        predicted_validity = disc_model(fake_fence)
                        target_validity = torch.ones(img_healthy.shape[0], 1) * -1
                        target_validity = target_validity.to(device)
                        # here the self['d_'] is fake_validity as we passed fake generated from z to disc
                        mae_validity = mae_criterion(predicted_validity, target_validity)
                        mse_recons = mse_criterion(fake_combined, img_tumor)
                        disjoincy = special_loss_disjoint_step1(fake_wild, fake_fence)

                        # g_loss = 0.5 * mae_validity + 0.1 * mse_recons + 0.4 * disjoincy
                        g_loss = mae_validity + mse_recons + 100 * disjoincy
                        g_loss.backward()
                        gen_optimizer.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    for _ in range(args.n_critic):
                        disc_optimizer.zero_grad()
                        # Generate a batch of fake images from tumor imgs
                        fake_wild, fake_fence, fake_combined = gen_model(img_tumor)
                        fake_labels = torch.ones(img_tumor.shape[0], 1).to(device)

                        # combine the fake fence and healthy img and label
                        combined_imgs = torch.cat((img_healthy, fake_fence), 0)
                        combined_labels = torch.cat((healthy_labels, fake_labels), 0)

                        # shuffle the combined imgs and labels
                        rand_perm_idx = torch.randperm(combined_imgs.shape[0]).to(device)
                        combined_imgs = torch.index_select(combined_imgs, 0, rand_perm_idx)
                        combined_labels = torch.index_select(combined_labels, 0, rand_perm_idx)

                        # train disc
                        predicted_validity = disc_model(combined_imgs)
                        
                        # Adversarial loss
                        d_loss = mae_criterion(predicted_validity, combined_labels)
                        # d_loss.backward()
                        # disc_optimizer.step()
                    
                    print(f"[Epoch {epoch}/{args.epoch}] \n"
                        f"[healthy_Batch {i}/{len(training_dataset_loader)}] \n"
                        f"[tumor_Batch {j}/{len(eval_dataset_loader)}] \n"
                        f"[tumor_SubBatch {idx}/{subbatch_len}] \n"
                        f"[D loss: {d_loss.item():5f}] \n"
                        f"[G loss: {g_loss.item():3f}] \n")

                    G_loss_mae_validity.append(mae_validity.item())
                    G_loss_mse_recons.append(mse_recons.item())
                    G_loss_disjoincy.append(disjoincy.item())
                    epoch_GLoss.append(g_loss.item())
                    epoch_DLoss.append(d_loss.item())

                    # if batches_done % (args.sample_interval * len(training_dataset_loader) * len(eval_dataset_loader)) == 0:
                    if(epoch % args.sample_interval == 0):
                        out = torch.cat(
                            (img_tumor[-5:].cpu(), fake_wild[-5:].cpu() ,fake_fence[-5:].cpu(), fake_combined[-5:].cpu())
                        )
                        plt.title(f'inference slice result')
                        plt.rcParams['figure.dpi'] = 256
                        plt.grid(False)
                        plt.imshow(gridify_output(out, 5), cmap='gray')
                        plt.savefig(f"./train_data_out/anchor/{batches_done:08}.png")
                    batches_done += 1
                    break
                break
            break
        # print epoch losses
        log_writer.add_scalar('G_loss_mae_validity', np.average(np.array(G_loss_mae_validity)), epoch)
        log_writer.add_scalar('G_loss_mse_recons', np.average(np.array(G_loss_mse_recons)), epoch)
        log_writer.add_scalar('G_loss_disjoincy', np.average(np.array(G_loss_disjoincy)), epoch)
        log_writer.add_scalar('epoch_GLoss', np.average(np.array(epoch_GLoss)), epoch)
        log_writer.add_scalar('epoch_DLoss', np.average(np.array(epoch_DLoss)), epoch)
        progress.update(1)
        if(epoch % args.save_weight_interval ==0):
                print("saving weight....") 
                weight_path = './weight/Generator%d.pt' %(int(epoch))
                torch.save(gen_model.module.state_dict(), weight_path)
                weight_path = './weight/Discriminator%d.pt' %(int(epoch))
                torch.save(disc_model.module.state_dict(), weight_path)

                weight_path = './mod_weight/Generator%d.pt' %(int(epoch))
                torch.save(gen_model.state_dict(), weight_path)
                weight_path = './mod_weight/Discriminator%d.pt' %(int(epoch))
                torch.save(disc_model.state_dict(), weight_path)
        torch.cuda.empty_cache()
