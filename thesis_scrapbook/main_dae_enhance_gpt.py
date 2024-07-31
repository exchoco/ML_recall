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
from model_dae_enhance import (VAE_withCrossSkip128skipMiddleLossDisent_Unroll, 
                                TripletNet, VAE_withCrossSkip128skipMiddleLoss_Unroll, 
                                VAE_withCrossSkip128skipMiddleLossConv_Unroll, 
                                VAE_withCrossSkipMiddleLossConv_Unroll, 
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll, 
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_concat,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_noskip,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_vvae,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_maxpool,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS)
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
    parser.add_argument("-save_weights" , type = bool , default = False)

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
    net = VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll()
    siam_net = TripletNet(net)
    siam_net = nn.DataParallel(siam_net, device_ids=[0,1,3])

    # siam_net.load_state_dict(torch.load("./mod_weight/TriVAE650.pt"))
    # start_epoch = 650
    siam_net.to(device)

    optimizer = torch.optim.Adam(siam_net.parameters(), lr=args.Learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)

    scheduler.step(start_epoch)
    # mse_criterion = nn.MSELoss(reduction = "sum")
    mae_criterion = nn.L1Loss(reduction = "sum")
    mae_criterion.cuda()

    mae_criterion_64 = nn.L1Loss(reduction = "sum")
    mae_criterion_64.cuda()

    triplet_margin = 1
    triplet_criterion = TripletLoss(triplet_margin)
    triplet_criterion.cuda()

    SSIM_loss = pytorch_ssim.SSIM()
    SSIM_loss.cuda()

    TV_criterion = TVLoss()
    TV_criterion.cuda()

    triplet_loss = (nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
    triplet_loss.cuda()

    t_log_dir = "./Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))
    tfr = 0
    progress = tqdm(total=args.epoch)
    for epoch in range(start_epoch, start_epoch + args.epoch + 1):
        siam_net.train()
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
        epoch_MMSE_anchor = []
        epoch_MMSE_pos = []
        epoch_MMSE_neg = []
        epoch_MMSE_anchor_64 = []
        epoch_MMSE_pos_64 = []
        epoch_MMSE_neg_64 = []
        batch = tqdm(training_dataset_loader,ncols=80)
        for idx, x in enumerate(batch):
            # x = x.float().cuda()
            for key, val in x.items():
                x[key] = val.to(device)

            # train the siamese encoders with triplet every iteration
            optimizer.zero_grad()

            # # assigning a new negative slices here to test out the coarse noise 
            x["noisy_anchor"] = add_coarse_noise(x["anchor"].clone())
            
            out_anchor, out_pos, out_neg = siam_net(x["negative_ori"], x["positive"], x["noisy_anchor"])
            enc_triplet_loss = triplet_criterion(out_anchor[4], out_pos[4], out_neg[4])

            rec_anchor_64 = out_anchor[5]
            rec_pos_64 = out_pos[5]
            rec_neg_64 = out_neg[5]
            
            rec_anchor, rec_pos, rec_neg = out_anchor[0], out_pos[0], out_neg[0]

            # mae_anchor = mae_criterion(rec_anchor, x["negative_ori"])
            # mae_pos = mae_criterion(rec_pos, x["positive"])
            mae_neg_to_ori = mae_criterion(rec_neg, x["anchor"])
            # mae_neg_to_noisy = mae_criterion(rec_neg, x["noisy_anchor"])

            kld_anchor = kl_criterion(out_anchor[2], out_anchor[3], args.batch_size)
            kld_pos = kl_criterion(out_pos[2], out_pos[3], args.batch_size)
            # kld_neg = kl_criterion(out_neg[2], out_neg[3], args.batch_size)

            # TV_anchor = TV_criterion(torch.abs(torch.sub(rec_anchor, x["anchor"])))
            # TV_pos = TV_criterion(torch.abs(torch.sub(rec_pos, x["positive"])))
            # TV_neg = TV_criterion(torch.abs(torch.sub(rec_neg, x["negative_ori"])))

            # SSIM_anchor = SSIM_loss(rec_anchor, x["negative_ori"])
            # SSIM_pos = SSIM_loss(rec_pos, x["positive"])
            SSIM_neg = SSIM_loss(rec_neg, x["anchor"])

             # loss for the 64 size
            resize_64 = torchvision.transforms.Resize([32, 32])
            x_anchor_64 = resize_64(x["negative_ori"])
            x_pos_64 = resize_64(x["positive"])
            x_neg_ori_64 = resize_64(x["anchor"])
            # x_neg_noisy_64 = resize_64(x["noisy_anchor"])
            # print(f"this resized gt {x_anchor_64.shape}")
            mae_anchor_64 = mae_criterion_64(rec_anchor_64, x_anchor_64)
            mae_pos_64 = mae_criterion_64(rec_pos_64, x_pos_64)
            mae_neg_to_ori_64 = mae_criterion_64(rec_neg_64, x_neg_ori_64)
            # mae_neg_to_noisy_64 = mae_criterion_64(rec_neg_64, x_neg_noisy_64)

            MMSE_anchor_64 = masked_mse(x_anchor_64, rec_anchor_64)
            MMSE_pos_64 = masked_mse(x_pos_64, rec_pos_64)
            MMSE_neg_64 = masked_mse(x_neg_ori_64, rec_neg_64)


            MMSE_anchor = masked_mse(x["negative_ori"], rec_anchor)
            MMSE_pos = masked_mse(x["positive"], rec_pos)
            MMSE_neg = masked_mse(x["anchor"], rec_neg)

            # enc_dec_loss = (MMSE_anchor + MMSE_pos + MMSE_neg) + (kld_anchor + kld_pos) + enc_triplet_loss - (SSIM_anchor + SSIM_pos + SSIM_neg)
            # enc_dec_loss = (0.05 * mae_anchor + 0.05 * mae_pos + 0.9 * mae_neg_to_ori + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_anchor + kld_pos + kld_neg) + enc_triplet_loss - (SSIM_anchor + SSIM_pos + SSIM_neg)
            # enc_dec_loss = (mae_anchor + mae_pos + mae_neg_to_ori + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_anchor + kld_pos + kld_neg) + enc_triplet_loss - (SSIM_anchor + SSIM_pos + SSIM_neg)
            # enc_dec_loss = (mae_anchor + mae_pos + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_neg) + enc_triplet_loss - (SSIM_anchor + SSIM_pos)

            # current best
            enc_dec_loss = (mae_neg_to_ori + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_anchor + kld_pos) + enc_triplet_loss - (SSIM_neg)
            # Tvae_c_nossim
            # enc_dec_loss = (mae_neg_to_ori + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_anchor + kld_pos) + enc_triplet_loss
            # Tvae_c_nossim_nocross and Tvae_c_nossim_noskip
            # enc_dec_loss = (mae_neg_to_ori) + (kld_anchor + kld_pos) + enc_triplet_loss
            # Tvae_c_nossim_vvae
            # enc_dec_loss = (mae_neg_to_ori) + (kld_neg)
            # Tvae_c_allskipunet
            # enc_dec_loss = (mae_neg_to_ori) - (SSIM_neg)
            # Tvae_c_unetWithGCS
            # enc_dec_loss = (mae_neg_to_ori + mae_neg_to_ori_64)
            # best but without midloss
            # enc_dec_loss = (mae_neg_to_ori) + (kld_anchor + kld_pos) + enc_triplet_loss - (SSIM_neg)


            # enc_dec_loss = (mae_anchor + mae_pos + mae_neg_to_ori + mae_anchor_64 + mae_pos_64 + mae_neg_to_ori_64) + (kld_anchor + kld_pos) + enc_triplet_loss - (SSIM_anchor + SSIM_pos + SSIM_neg)
            # enc_dec_loss = (MMSE_neg + MMSE_anchor_64 + MMSE_pos_64 + MMSE_neg_64) + (kld_anchor + kld_pos) - (SSIM_neg)

            enc_dec_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(siam_net.parameters(), max_norm=0.5)
            optimizer.step()

            # save the training recontructed images for every 5th epoch and index == 0
            if(int(epoch) % args.save_weight_interval == 0 and idx == 0):
                print("saving training images samples")
                upsample = nn.Upsample(scale_factor=8)
                create_folder_save_tensor_imgs("train_data_out/anchor", sample_count, f"train_data_epoch{epoch}.png", x["negative_ori"][-sample_count:, ...] , rec_anchor[-sample_count:, ...])
                create_folder_save_tensor_imgs("train_data_out/positive", sample_count, f"train_data_epoch{epoch}.png", x["positive"][-sample_count:, ...] , rec_pos[-sample_count:, ...])
                create_folder_save_tensor_imgs("train_data_out/negative", sample_count, f"train_data_epoch{epoch}.png", x["noisy_anchor"][-sample_count:, ...] , rec_neg[-sample_count:, ...])

                create_folder_save_tensor_imgs("train_data_out/anchor_64", sample_count, f"train_data_epoch{epoch}.png", x["negative_ori"][-sample_count:, ...] , upsample(rec_anchor_64)[-sample_count:, ...])
                create_folder_save_tensor_imgs("train_data_out/positive_64", sample_count, f"train_data_epoch{epoch}.png", x["positive"][-sample_count:, ...] , upsample(rec_pos_64)[-sample_count:, ...])
                create_folder_save_tensor_imgs("train_data_out/negative_64", sample_count, f"train_data_epoch{epoch}.png", x["noisy_anchor"][-sample_count:, ...] , upsample(rec_neg_64)[-sample_count:, ...])

            epoch_enc_triplet_loss.append(enc_triplet_loss.detach().cpu().numpy())
            epoch_enc_dec_loss.append(enc_dec_loss.detach().cpu().numpy())
            # epoch_mae_anchor.append(mae_anchor.detach().cpu().numpy())
            # epoch_mae_pos.append(mae_pos.detach().cpu().numpy())
            epoch_mae_neg_to_ori.append(mae_neg_to_ori.detach().cpu().numpy())
            # epoch_mae_neg_to_noisy.append(mae_neg_to_noisy.detach().cpu().numpy())
            epoch_kld_anchor.append(kld_anchor.detach().cpu().numpy())
            epoch_kld_pos.append(kld_pos.detach().cpu().numpy())
            # epoch_kld_neg.append(kld_neg.detach().cpu().numpy())
            # epoch_TV_anchor.append(TV_anchor.detach().cpu().numpy())
            # epoch_TV_pos.append(TV_pos.detach().cpu().numpy())
            # epoch_TV_neg.append(TV_neg.detach().cpu().numpy())
            # epoch_SSIM_anchor.append(SSIM_anchor.detach().cpu().numpy())
            # epoch_SSIM_pos.append(SSIM_pos.detach().cpu().numpy())
            epoch_SSIM_neg.append(SSIM_neg.detach().cpu().numpy())
            epoch_MMSE_anchor.append(MMSE_anchor.detach().cpu().numpy())
            epoch_MMSE_pos.append(MMSE_pos.detach().cpu().numpy())
            epoch_MMSE_neg.append(MMSE_neg.detach().cpu().numpy())
            epoch_MMSE_anchor_64.append(MMSE_anchor_64.detach().cpu().numpy())
            epoch_MMSE_pos_64.append(MMSE_pos_64.detach().cpu().numpy())
            epoch_MMSE_neg_64.append(MMSE_neg_64.detach().cpu().numpy())

        scheduler.step()
        
        log_writer.add_scalar('epoch_enc_triplet_loss', np.mean(np.array(epoch_enc_triplet_loss)), epoch)
        log_writer.add_scalar('epoch_enc_dec_loss', np.mean(np.array(epoch_enc_dec_loss)), epoch)

        # log_writer.add_scalar('epoch_mae_anchor', np.mean(np.array(epoch_mae_anchor)), epoch)
        # log_writer.add_scalar('epoch_mae_pos', np.mean(np.array(epoch_mae_pos)), epoch)
        log_writer.add_scalar('epoch_mae_neg_to_ori', np.mean(np.array(epoch_mae_neg_to_ori)), epoch)
        # log_writer.add_scalar('epoch_mae_neg_to_noisy', np.mean(np.array(epoch_mae_neg_to_noisy)), epoch)

        log_writer.add_scalar('epoch_kld_anchor', np.mean(np.array(epoch_kld_anchor)), epoch)
        log_writer.add_scalar('epoch_kld_pos', np.mean(np.array(epoch_kld_pos)), epoch)
        # log_writer.add_scalar('epoch_kld_neg', np.mean(np.array(epoch_kld_neg)), epoch)

        # log_writer.add_scalar('epoch_TV_anchor', np.mean(np.array(epoch_TV_anchor)), epoch)
        # log_writer.add_scalar('epoch_TV_pos', np.mean(np.array(epoch_TV_pos)), epoch)
        # log_writer.add_scalar('epoch_TV_neg', np.mean(np.array(epoch_TV_neg)), epoch)

        # log_writer.add_scalar('epoch_SSIM_anchor', np.mean(np.array(epoch_SSIM_anchor)), epoch)
        # log_writer.add_scalar('epoch_SSIM_pos', np.mean(np.array(epoch_SSIM_pos)), epoch)
        log_writer.add_scalar('epoch_SSIM_neg', np.mean(np.array(epoch_SSIM_neg)), epoch)

        log_writer.add_scalar('epoch_MMSE_anchor', np.mean(np.array(epoch_MMSE_anchor)), epoch)
        log_writer.add_scalar('epoch_MMSE_pos', np.mean(np.array(epoch_MMSE_pos)), epoch)
        log_writer.add_scalar('epoch_MMSE_neg', np.mean(np.array(epoch_MMSE_neg)), epoch)

        log_writer.add_scalar('epoch_MMSE_anchor_64', np.mean(np.array(epoch_MMSE_anchor_64)), epoch)
        log_writer.add_scalar('epoch_MMSE_pos_64', np.mean(np.array(epoch_MMSE_pos_64)), epoch)
        log_writer.add_scalar('epoch_MMSE_neg_64', np.mean(np.array(epoch_MMSE_neg_64)), epoch)

        print(f"epoch {epoch} \n")
        print("epoch encoder triplet loss: ", np.mean(np.array(epoch_enc_triplet_loss)))
        print("\nepoch encoder decoder l1 and kld loss: ", np.mean(np.array(epoch_enc_dec_loss)))

        progress.update(1)
        if(int(epoch) % args.save_weight_interval == 0):
            features = None
            labels = None
            # run eval and save evaluation image here
            with torch.no_grad():
                print("saving evaluation images samples")
                siam_net.eval()
                for x in testing_dataset_loader:
                    for key, val in x.items():
                        x[key] = val.to(device)
                
                # # train the siamese encoders with triplet every iteration
                x["noisy_anchor"] = add_coarse_noise(x["anchor"].clone())
                
                out_anchor, out_pos, out_neg = siam_net(x["negative_ori"], x["positive"], x["noisy_anchor"])
                rec_anchor, rec_pos, rec_neg = out_anchor[0], out_pos[0], out_neg[0]
                upsample = nn.Upsample(scale_factor=8)
                rec_anchor_64 = out_anchor[5]
                rec_pos_64 = out_pos[5]
                rec_neg_64 = out_neg[5]

                create_folder_save_tensor_imgs("eval_data_out/anchor", sample_count, f"eval_data_epoch{epoch}.png", x["negative_ori"][-sample_count:, ...] , rec_anchor[-sample_count:, ...])
                create_folder_save_tensor_imgs("eval_data_out/positive", sample_count, f"eval_data_epoch{epoch}.png", x["positive"][-sample_count:, ...] , rec_pos[-sample_count:, ...])
                create_folder_save_tensor_imgs("eval_data_out/negative", sample_count, f"eval_data_epoch{epoch}.png", x["noisy_anchor"][-sample_count:, ...] , rec_neg[-sample_count:, ...])
                
                create_folder_save_tensor_imgs("eval_data_out/anchor_64", sample_count, f"eval_data_epoch{epoch}.png", x["negative_ori"][-sample_count:, ...] , upsample(rec_anchor_64)[-sample_count:, ...])
                create_folder_save_tensor_imgs("eval_data_out/positive_64", sample_count, f"eval_data_epoch{epoch}.png", x["positive"][-sample_count:, ...] , upsample(rec_pos_64)[-sample_count:, ...])
                create_folder_save_tensor_imgs("eval_data_out/negative_64", sample_count, f"eval_data_epoch{epoch}.png", x["noisy_anchor"][-sample_count:, ...] , upsample(rec_neg_64)[-sample_count:, ...])


                # this is to infer on noisy and normal anchor, and plot their TSNE
                if features is not None:
                    features = torch.cat((features, out_neg[4].view(out_neg[4].size(0), -1)))
                else:
                    features = out_anchor[4].view(out_neg[4].size(0), -1)
                
                features = torch.cat((features, out_neg[4].view(out_neg[4].size(0), -1)))


            # besides of the evaluation, lets do direct inference and see the performance
            # if(int(epoch) > 500 and int(epoch) % 200 == 0 ):
            #     dice_data = []
            #     AUC_scores = []
            #     intersection_list = []
            #     union_list = []
            #     with torch.no_grad():
            #         siam_enc.eval()
            #         siam_dec.eval()
            #         print("calculating anomalous inference DICE and AUROC")
            #         for idx, x in enumerate(eval_dataset_loader):
            #             x["image"] = x["image"].to(device)
            #             x["mask"] = x["mask"].to(device)

            #             # solve the shape since we dont need the batch dimension
            #             x["image"] = x["image"].view(-1, 1, x["image"].shape[-2], x["image"].shape[-2])
            #             x["mask"] = x["mask"].view(-1, 1, x["mask"].shape[-2], x["mask"].shape[-2])

            #             # out_anchor, out_pos, out_neg = siam_enc.(x["image"], x["image"], x["image"])
            #             # rec_anchor, rec_pos, rec_neg = siam_dec(out_anchor, out_pos, out_neg)

            #             out_anchor = siam_enc.module.get_embedding(x["image"])
            #             rec_anchor= siam_dec.module.get_embedding(out_anchor)
                        
            #             # # since we are already outside of the inference, we dont need the 2nd dimension anymore
            #             # x["image"] = x["image"].view(-1, x["image"].shape[-2], x["image"].shape[-2])
            #             # x["mask"] = x["mask"].view(-1, x["mask"].shape[-2], x["mask"].shape[-2])
            #             # rec_anchor = rec_anchor.view(-1, rec_anchor.shape[-2], rec_anchor.shape[-1])

            #             image_mask = np.zeros(np.array(x["image"].cpu()).shape)
            #             image_mask[np.array(x["image"].cpu())>0] = 1
            #             image_mask = torch.from_numpy(image_mask).to(device)

            #             residual = abs(x["image"] - rec_anchor) * image_mask

            #             fpr_simplex, tpr_simplex, _ = ROC_AUC(x["mask"].to(torch.uint8), residual)
            #             AUC_scores.append(AUC_score(fpr_simplex, tpr_simplex))
            #             # print(f"======this is AUROC score {AUC_scores[-1]}")

            #             residual_backup = residual
            #             residual = (residual > args.thresh).float()

            #             # do post processing on the residual
            #             # print(f"this is residual shape {residual.shape}")
            #             residual = residual.view(-1, residual.shape[-2], residual.shape[-1])
            #             residual = apply_3d_median_filter(residual.cpu())
            #             residual = filter_3d_connected_components(residual)
            #             residual = torch.from_numpy(residual.reshape(-1, 1, residual.shape[-2], residual.shape[-1])).to(device)
                        
            #             # show samples
            #             out = torch.cat(
            #             (x["image"][40:50].cpu(), rec_anchor[40:50].cpu() ,residual_backup[40:50].cpu(), residual[40:50].cpu(), x["mask"][40:50].cpu())
            #             )
            #             # plt.title(f'inference slice result')
            #             # plt.rcParams['figure.dpi'] = 150
            #             # plt.grid(False)
            #             # plt.imshow(gridify_output(out, 10), cmap='gray')
            #             # plt.show()

            #             subject_dice, intersection, union = dice_coeff(
            #                         x["image"], rec_anchor,
            #                         x["mask"], mse=residual
            #                         )
            #             dice_data.append(subject_dice.cpu().item())
            #             intersection_list.append(intersection.cpu().item())
            #             union_list.append(union.cpu().item())
            #             # print(f"======this is subject DICE score {subject_dice}")


                    
            #         dice = 2 * np.sum(intersection_list) / np.sum(union_list)
            #         print(f"OVERALL Dice coefficient: {dice}")
            #         print(f"OVERALL AUROC : {np.nanmean(AUC_scores)}")

            #         log_writer.add_scalar('dataset_dice', dice, epoch)
            #         log_writer.add_scalar('average_patient_dice', np.mean(dice_data), epoch)
            #         log_writer.add_scalar('average_AUROC', np.nanmean(AUC_scores), epoch)

            if(int(epoch) >= 900 or int(epoch) == 0 and args.save_weights):
                print("saving model weights .... ")
                # weight_path = './weight/TriVAE%d.pt' %(int(epoch))
                # torch.save(siam_net.module.state_dict(), weight_path)

                weight_path = './mod_weight/TriVAE%d.pt' %(int(epoch))
                torch.save(siam_net.state_dict(), weight_path)

            torch.cuda.empty_cache()