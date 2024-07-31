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
    CNN_EncoderNoAtt,
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
    CNN_Decoder_SpatialCrossConcatenate3OnlyMiddleLossWith128Skip)
from utils import *
from sklearn.manifold import TSNE

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size" , type = int , default = 36)
    # parser.add_argument("-batch_size" , type = int , default = 60)
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
    Encoder = CNN_EncoderWithCoorAvgPool()
    Decoder = CNN_Decoder_SpatialCrossSkip3OnlyMiddleLoss()
    siam_enc = TripletNet(Encoder)
    siam_dec = TripletNet(Decoder)
    siam_enc = nn.DataParallel(siam_enc)
    siam_dec = nn.DataParallel(siam_dec)
    siam_enc.to(device)
    siam_enc.load_state_dict(torch.load("./mod_weight/TriEncoder750.pt"))
    siam_dec.to(device)
    siam_dec.load_state_dict(torch.load("./mod_weight/TriDecoder750.pt"))


    dice_data = []
    AUC_scores = []
    intersection_list = []
    union_list = []
    adaptive_total_AUROC = []
    total_slices = 0

    feature_slice = 13

    features = None
    labels = None

    siam_enc.eval()
    siam_dec.eval()
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

                    out_anchor = siam_enc.module.get_embedding(x["image"])
                    rec_anchor = siam_dec.module.get_embedding(out_anchor)
                    rec_anchor_64 = rec_anchor[1]
                    rec_anchor = rec_anchor[0]
                    # rec_anchor, temp, temp2 = siam_dec.module.get_embedding(out_anchor)

                    # # manage the skip featres for visualization
                    # upsample = nn.Upsample(scale_factor=4)
                    # temp = upsample(temp)
                    # # temp = temp[:, feature_slice].view(-1, 1, 256, 256)
                    # temp = torch.mean(temp, dim = [1]).view(-1, 1, 256, 256)

                    # upsample = nn.Upsample(scale_factor=4)
                    # temp2 = upsample(temp2)
                    # # temp2 = temp2[:, feature_slice].view(-1, 1, 256, 256)
                    # temp2 = torch.mean(temp2, dim = [1]).view(-1, 1, 256, 256)
                    # print(f"this is temp shape {temp.shape}")
                    
                    # features are saved for the TSNE plot later
                    if features is not None:
                        features = torch.cat((features, out_anchor[0].view(-1, out_anchor[0].shape[-1])))
                    else:
                        features = out_anchor[0].view(-1, out_anchor[0].shape[-1])
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

                    fpr_simplex, tpr_simplex, _ = ROC_AUC(x["mask"].to(torch.uint8), residual)
                    AUC_scores.append(AUC_score(fpr_simplex, tpr_simplex))
                    print(f"======this is AUROC score {AUC_scores[-1]}")
                    if(not math.isnan(AUC_scores[-1])):
                        adaptive_total_AUROC.append(float(AUC_scores[-1]) * x["image"].shape[0])
                        total_slices+=x["image"].shape[0]

                    residual_backup = residual
                    residual = (residual > thresh).float()

                    # do post processing on the residual
                    # print(f"this is residual shape {residual.shape}")
                    residual = residual.view(-1, residual.shape[-2], residual.shape[-1])
                    residual = apply_3d_median_filter(residual.cpu())
                    residual = filter_3d_connected_components(residual)
                    residual = torch.from_numpy(residual.reshape(-1, 1, residual.shape[-2], residual.shape[-1])).to(device)
                    
                    # show samples
                    # if(x["image"].shape[0] > 50):
                    if(x["image"].shape[0] >= 36):
                        # tempshape = temp[40:50, [0]].shape
                        # print(f"image shape {tempshape}")
                        # print( rec_anchor[30:35].cpu())
                        out = torch.cat(
                        (x["image"][30:35].cpu(), rec_anchor[30:35].cpu() ,residual_backup[30:35].cpu(), residual[30:35].cpu(), x["mask"][30:35].cpu())

                        # (x["image"][40:45].cpu(), rec_anchor[40:45].cpu() ,residual_backup[40:45].cpu(), residual[40:45].cpu(), x["mask"][40:45].cpu())
                        # (x["image"][30:35].cpu(), rec_anchor[30:35].cpu() ,residual_backup[30:35].cpu(), residual[30:35].cpu(), x["mask"][30:35].cpu(),
                        # temp[30:35].cpu() * 255, temp2[30:35].cpu() * 255)
                        )
                        plt.title(f'inference slice result')
                        plt.rcParams['figure.dpi'] = 500
                        plt.grid(False)
                        plt.imshow(gridify_output(out, 5), cmap='gray')
                        # plt.savefig('./temp.png')
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
                            f.write(f"subj {idx} :\n")
                            f.write(f"subject dice: {subject_dice}\n")
                            f.write(f"AUROC: {AUC_scores[-1]}\n")
            print(f"Overall threshold {int(thresh * 100)}: ")
            dice = 2 * np.sum(intersection_list) / np.sum(union_list)
            print(f"dataset Dice coefficient: {dice}")
            print(f"average AUROC : {np.sum(adaptive_total_AUROC) / total_slices}")
            with open(f"./metrics/att3_overall_t{int(thresh * 100)}.csv", mode="w") as f:
                f.write(f"dataset dice: {dice}\n")
                f.write(f"average dice: {np.mean(dice_data)}\n")
                f.write(f"average AUROC: {np.sum(adaptive_total_AUROC) / total_slices}\n")
    
    tsne = TSNE(n_components=3, init='random', random_state=0, verbose=0, perplexity=30).fit_transform(features.cpu())
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(ty)

    visualize_tsne_points(tx, ty, labels)