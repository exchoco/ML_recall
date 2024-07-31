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
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_forshow, 
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_Spatial,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_concat,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_noskip,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_vvae,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_maxpool,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_allskipunet_forshow,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_CatSkipMidloss_forshow,
                                VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_unetWithGCS,
                                )
from utils import *
from sklearn.manifold import TSNE
import pytorch_ssim

binary_map = np.array([
    [0.        , 0, 0],
    [1.        , 0.       , 0]])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size" , type = int , default = 36)
    # parser.add_argument("-batch_size" , type = int , default = 60)
    parser.add_argument("-vae_restoration" , type = bool , default = False)
    parser.add_argument("-restoration_steps" , type = int , default = 10)
    parser.add_argument("-restoration_lr" , type = float , default = 1e-3)
    parser.add_argument("-restoration_TV_lambda" , type = float , default = 0)
    args = parser.parse_args()


    mkdir("./metrics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device:
        print ("-------------------------------------")
        print ("GPU is ready")
    
    DATASET_PATH = '../23_01_15_AnoDDPM/AnoDDPM/DATASETS/CancerousDataset/BraTS_T2'
    a_set = AnomalousMRIDataset(DATASET_PATH)
    eval_dataset_loader = torch.utils.data.DataLoader(
                    a_set,
                    batch_size= 1, shuffle=False,
                    num_workers=0, drop_last=True
                    )
    # Announce the model
    net = VAE_withCrossSkip128skipMiddleLossConv_Disent_Unroll_forshow()
    siam_net = TripletNet(net)
    siam_net = nn.DataParallel(siam_net, device_ids=[0,1,3])
    siam_net.load_state_dict(torch.load("./tvae_c_complete_att2/mod_weight/TriVAE1200.pt"))
    siam_net.to(device)

    dice_data = []
    AUC_scores = []
    AUPRC_scores = []
    intersection_list = []
    union_list = []
    adaptive_total_AUROC = []
    adaptive_total_AUPRC = []
    total_slices = 0

    feature_slice = 13

    features = None
    labels = None

    overall_idx = 0
    siam_net.eval()
    for thresh in [0.1] :
        with torch.no_grad():
            progress = tqdm(total=len(eval_dataset_loader))
            for subj_idx, x in enumerate(eval_dataset_loader):
                x["mask"] = x["mask"]
                x["image"] = x["image"]

                x["image"] = x["image"].to(device)
                x["mask"] = x["mask"].to(device)
                
                # solve the shape since we dont need the batch dimension
                x["image"] = x["image"].view(-1, 1, x["image"].shape[-2], x["image"].shape[-2])
                x["mask"] = x["mask"].view(-1, 1, x["mask"].shape[-2], x["mask"].shape[-2])

                # add noise before the denoising
                # x["noised_image"] = add_coarse_noise(x["image"].clone())

                # since dataset was batch size was not adjustable, fix here
                original_image = x["image"]
                original_mask = x["mask"]
                original_labels = x["label"]
                for idx in range(int(math.ceil(original_image.shape[0] / args.batch_size))):
                    if(original_image.shape[0] - 1 >= ((idx + 1) * args.batch_size) ):
                        start_idx = idx * args.batch_size
                        end_idx = (idx+1) * args.batch_size
                    else:
                        remaining = original_image.shape[0] % args.batch_size
                        start_idx = -remaining
                        end_idx = None

                    x["image"] = original_image[start_idx:end_idx]
                    # temp = x["image"].shape
                    # print(f"this is minibatch image shape {temp}")
                    x["mask"] = original_mask[start_idx:end_idx]
                    # temp = x["mask"].shape
                    # print(f"this is minibatch mask shape {temp}")
                    x["label"] = original_labels[0][start_idx:end_idx]
                    # print(f"this is label original shape {original_labels.shape}")
                    # temp = (x["label"]).shape
                    # print(f"this is minibatch label shape {temp}")
                    # out_anchor, out_pos, out_neg = siam_enc.(x["image"], x["image"], x["image"])
                    # rec_anchor, rec_pos, rec_neg = siam_dec(out_anchor, out_pos, out_neg)

                    if(args.vae_restoration):
                        with torch.enable_grad():
                            x_pred = x["image"].clone().requires_grad_(True)
                            for step in range(args.restoration_steps):
                                mae_criterion = nn.L1Loss(reduction = "sum")
                                mae_criterion.cuda()
                                TV_criterion = TVLoss()
                                TV_criterion.cuda()
                                SSIM_loss = pytorch_ssim.SSIM()
                                SSIM_loss.cuda()
                                out_anchor = siam_net.module.get_embedding(x["image"])
                                rec_anchor = out_anchor[0]

                                # debug, the rec was compared against the original not current restored, kld too
                                mae_anchor = mae_criterion(rec_anchor, x_pred)
                                kld_anchor = kl_criterion(out_anchor[2], out_anchor[3], x_pred.shape[0])
                                SSIM_anchor = SSIM_loss(rec_anchor, x_pred)

                                loss = mae_anchor + kld_anchor + SSIM_anchor
                                loss.retain_grad()
                                TV = TV_criterion(torch.sub(x["image"], rec_anchor))
                                TV.retain_grad()

                                # the create_graph was set to true previously, try use default
                                # gradients = torch.autograd.grad((loss + TV ), x_pred, create_graph=True)[0]
                                gradients = torch.autograd.grad((loss + args.restoration_TV_lambda * TV ), x_pred)[0]

                                x_pred = x_pred.cpu().detach() - args.restoration_lr * gradients.cpu().detach()
                                x_pred = x_pred.to(device).requires_grad_(True)
                        rec_anchor = x_pred
                    else:
                        out_anchor = siam_net.module.get_embedding(x["image"].clone())
                        rec_anchor = out_anchor[0]
                        forshow = out_anchor[-1]

                    # manage the skip features for visualization
                    upsample = nn.Upsample(scale_factor=4)
                    forshow = upsample(forshow)
                    # temp = temp[:, feature_slice].view(-1, 1, 256, 256)
                    # temp = torch.mean(temp, dim = [1]).view(-1, 1, 256, 256)
                    # temp = temp[:,0,:,:].view(-1, 1, 256, 256)
                    forshow = torch.mean(forshow, dim = [1]).view(-1, 1, 256, 256)

                    # upsample = nn.Upsample(scale_factor=4)
                    # temp2 = upsample(temp2)
                    # # temp2 = temp2[:, feature_slice].view(-1, 1, 256, 256)
                    # temp2 = torch.mean(temp2, dim = [1]).view(-1, 1, 256, 256)
                    # # temp2 = F.normalize(temp2, p=2, dim = 1)
                    # print(f"this is temp shape {temp.shape}")
                    
                    # # manage the attention map visualization
                    # upsample = nn.Upsample(scale_factor=4)
                    # forshow = forshow.view(forshow.shape[0], -1, 64, 64)
                    # forshow = torch.mean(forshow, dim = [1]).view(-1, 1, 64, 64)
                    # forshow = upsample(forshow)
                    # print(f"the forshow shape {forshow.shape}")
                    

                    # features are saved for the TSNE plot later
                    # if features is not None:
                    #     features = torch.cat((features, out_anchor[1].view(-1, out_anchor[1].shape[-1])))
                    # else:
                    #     features = out_anchor[1].view(-1, out_anchor[1].shape[-1])

                    # use the following since the feature we triplet is the one before sampling is done
                    if features is not None:
                        features = torch.cat((features, out_anchor[4].view(out_anchor[4].size(0), -1)))
                    else:
                        features = out_anchor[4].view(out_anchor[4].size(0), -1)

                    print(f"this is feature shape {features.shape}")
                    if labels is not None:
                        labels = np.concatenate((labels, x["label"].T))
                    else:
                        labels = x["label"].T
                    print(f"this is label shape {labels.shape}")
                    # # since we are already outside of the inference, we dont need the 2nd dimension anymore
                    # x["image"] = x["image"].view(-1, x["image"].shape[-2], x["image"].shape[-2])
                    # x["mask"] = x["mask"].view(-1, x["mask"].shape[-2], x["mask"].shape[-2])
                    # rec_anchor = rec_anchor.view(-1, rec_anchor.shape[-2], rec_anchor.shape[-1])

                    image_mask = np.zeros(np.array(x["image"].cpu()).shape)
                    image_mask[np.array(x["image"].cpu())>0] = 1
                    image_mask = torch.from_numpy(image_mask).to(device)

                    residual = abs(x["image"] - rec_anchor) * image_mask

                    # # try to show the histogram
                    # hist = torch.histc(residual, 10, min=0, max=1)
                    
                    # for idx, x in enumerate(np.array(hist.cpu())):
                    #     print(f"{idx} : {x}")
                    # exit()

                    fpr_simplex, tpr_simplex, _ = ROC_AUC(x["mask"].to(torch.uint8), residual)
                    precision, recall, _ = ROC_PRC(x["mask"].to(torch.uint8), residual)
                    AUC_scores.append(AUC_score(fpr_simplex, tpr_simplex))
                    AUPRC_scores.append(AUC_score(recall, precision))
                    print(f"======this is AUROC score {AUC_scores[-1]}")
                    print(f"======this is AURPRC score {AUPRC_scores[-1]}")
                    if(not math.isnan(AUC_scores[-1])):
                        adaptive_total_AUROC.append(float(AUC_scores[-1]) * x["image"].shape[0])
                        # adaptive_total_AUPRC.append(float(AUPRC_scores[-1]) * x["image"].shape[0])
                        total_slices+=x["image"].shape[0]
                    adaptive_total_AUPRC.append(float(AUPRC_scores[-1]))
                    residual_backup = residual
                    residual = (residual > thresh).float()
                    
                    # do post processing on the residual
                    # print(f"this is residual shape {residual.shape}")
                    residual = residual.view(-1, residual.shape[-2], residual.shape[-1])
                    residual = apply_3d_median_filter(residual.cpu())
                    residual = filter_3d_connected_components(residual)
                    residual = torch.from_numpy(residual.reshape(-1, 1, residual.shape[-2], residual.shape[-1])).to(device)


                    # show samples
                    if(x["image"].shape[0] > 35):
                    # if(x["image"].shape[0] > 36):
                        # tempshape = temp[40:50, [0]].shape
                        # print(f"image shape {tempshape}")
                        # print( rec_anchor[30:35].cpu())
                        start_slice = 27
                        end_slice = 28
                        red_residuals = residual[start_slice:end_slice].cpu().repeat(1, 3, 1, 1)
                        red_residuals[:, 1:, : ,:] = 0
                        out = torch.cat(
                        # (x["image"][40:45].cpu(), rec_anchor[40:45].cpu() ,residual_backup[40:45].cpu(), residual[40:45].cpu(), x["mask"][40:45].cpu())
                        # (x["image"][30:35].cpu(), rec_anchor[30:35].cpu() ,residual_backup[30:35].cpu(), residual[30:35].cpu(), x["mask"][30:35].cpu(),
                        # temp2[30:35].cpu(), temp[30:35].cpu())

                        (x["image"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1), rec_anchor[start_slice:end_slice].cpu().repeat(1, 3, 1, 1) ,
                        residual_backup[start_slice:end_slice].cpu().repeat(1, 3, 1, 1), red_residuals + x["image"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1), 
                        x["mask"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1), forshow[start_slice:end_slice].cpu().repeat(1, 3, 1, 1))

                        # (x["image"][23:28].cpu(), rec_anchor[23:28].cpu() ,forshow[23:28].cpu(), residual_backup[23:28].cpu(), residual[23:28].cpu(), x["mask"][23:28].cpu())

                        )
                        plt.title(f'inference slice result')
                        plt.rcParams['figure.dpi'] = 1024
                        plt.grid(False)
                        plt.imshow(gridify_output(out, 5), cmap='gray')
                        # plt.imshow(((residual_backup[start_slice:end_slice].cpu())[0][0]* 255).clamp(0, 255).to(torch.uint8), cmap='gray')
                        # plt.imshow((rec_anchor[start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                        # plt.savefig('./temp.png')
                        print(f"the overall id of current sample {overall_idx}")
                        if(overall_idx == 1):
                            plt.imshow((x["image"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./input.png')
                            plt.imshow((x["mask"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./gt.png')
                            plt.imshow((rec_anchor[start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./rec.png')
                            plt.imshow((red_residuals + x["image"][start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./output.png')
                            plt.imshow((residual_backup[start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./residual.png')
                            # plt.savefig('./temp.png')

                            # temp
                            plt.imshow(residual[start_slice:end_slice].cpu()[0][0], cmap='gray')
                            plt.savefig('./residual_complete.png')
                            # plt.imshow(residual_tres[start_slice:end_slice].cpu()[0][0], cmap='gray')
                            # plt.savefig('./residual_tres.png')
                            # plt.imshow(residual_tres_med_conn[start_slice:end_slice].cpu()[0][0], cmap='gray')
                            # plt.savefig('./residual_tres_med_conn.png')
                            # plt.imshow(residual_tres_med[start_slice:end_slice].cpu()[0][0], cmap='gray')
                            # plt.savefig('./residual_tres_med.png')

                            plt.imshow((forshow[start_slice:end_slice].cpu().repeat(1, 3, 1, 1))[0].permute(1,2,0))
                            plt.savefig('./forshow.png')

                            exit()
                        # plt.show()

                    subject_dice, intersection, union = dice_coeff(
                                x["image"], rec_anchor,
                                x["mask"], mse=residual
                                )
                    dice_data.append(subject_dice.cpu().item())
                    intersection_list.append(intersection.cpu().item())
                    union_list.append(union.cpu().item())
                    print(f"======this is subject DICE score {subject_dice}")
                    with open(f"./metrics/att3_record_t{int(thresh * 100)}.csv", mode="a") as f:
                            f.write(f"subj {overall_idx} :\n")
                            f.write(f"subject dice: {subject_dice}\n")
                            f.write(f"AUROC: {AUC_scores[-1]}\n")
                    overall_idx += 1
            print(f"Overall threshold {int(thresh * 100)}: ")
            dice = 2 * np.sum(intersection_list) / np.sum(union_list)
            print(f"dataset Dice coefficient: {dice}")
            # this AUROC calculation is the same as a direct nan mean
            print(f"average AUROC : {np.sum(adaptive_total_AUROC) / total_slices}")
            print(f"average AUPRC : {np.mean(adaptive_total_AUPRC)}")
            with open(f"./metrics/att3_overall_t{int(thresh * 100)}.csv", mode="w") as f:
                f.write(f"dataset dice: {dice}\n")
                f.write(f"average dice: {np.mean(dice_data)}\n")
                f.write(f"average AUROC: {np.sum(adaptive_total_AUROC) / total_slices}\n")
                f.write(f"average AUPRC : {np.mean(adaptive_total_AUPRC)}\n")

    
    tsne = TSNE(n_components=3, init='random', random_state=0, verbose=0, perplexity=30).fit_transform(features.cpu())
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(ty)

    visualize_tsne_points(tx, ty, labels)