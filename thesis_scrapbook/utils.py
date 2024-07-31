from inspect import ismemberdescriptor
from re import sub
from tkinter.messagebox import NO
from PIL import Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from imageio import imread
import pandas as pd
import argparse
from numpy import append
import re
from simplex import Simplex_CLASS
import torchvision.utils
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from skimage.measure import regionprops, label
import scipy
from sklearn.manifold import TSNE
import random

colors_per_class = {
    0 : [255, 107, 107],
    1 : [254, 202, 87],
    # 2 : [10, 189, 227],
    # 3 : [255, 159, 243],
    # 4 : [16, 172, 132],
    # 5 : [128, 80, 128],
    # 6 : [87, 101, 116],
    # 7 : [52, 31, 151],
    # 8 : [0, 0, 0],
    # 9 : [100, 100, 255],
}

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(path+' Create Success')
    else:
        print(path+' Already exist')


default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])
def tryint(s):
    try:
        return int(s)
    except:
        return s

def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)

def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas

def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
        ):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise

def show_single_image(image1, image2):
    # show image for checking
    print(f"this is npy shape {image1.shape}")
    # image_im = (image1[0] *255).to(torch.uint8)
    image_im = (image1[0] *255).clamp(0, 255).to(torch.uint8)
    image_arr = np.array(image_im.cpu(), dtype=np.uint8)

    # image_im2 = (image2[0] *255).to(torch.uint8)
    image_im2 = (image2[0] *255).clamp(0, 255).to(torch.uint8)
    image_arr2 = np.array(image_im2.cpu(), dtype=np.uint8)


    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 2, 1)
    plt.imshow(  image_arr, cmap="gray" )
    fig.add_subplot(2, 2, 2)
    plt.imshow(  image_arr2, cmap="gray" )
    plt.show()

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def kl_criterion(mu, logvar, batchsize):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batchsize
    return KLD


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletLoss_CosineDist(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1):
        super(TripletLoss_CosineDist, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        similarity_positive = torch.cosine_similarity(anchor.squeeze(1), positive.squeeze(1))
        similarity_negative = torch.cosine_similarity(anchor.squeeze(1), negative.squeeze(1))
        loss = torch.maximum(similarity_positive - similarity_negative + torch.tensor(self.margin).type(torch.float), torch.tensor(0).type(torch.float))
        return loss.mean() if size_average else loss.sum()

def gridify_output(img, row_size=-1):
    # scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    scale_img = lambda img: (img * 255).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )

def create_folder_save_tensor_imgs(folder_name, row_size, image_name, tensor1, tensor2):
    # this function create specified folder and save tensor img
    mkdir(folder_name)
    out2 = torch.cat(
                (tensor1.cpu(), tensor2.cpu())
                )
    plt.title(image_name)

    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    plt.imshow(gridify_output(out2, row_size), cmap='gray')

    plt.savefig(f'{folder_name}/{image_name}')
    plt.clf()

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())

def ROC_PRC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return precision_recall_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return precision_recall_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

def apply_3d_median_filter(volume, kernelsize=5):
    volume = scipy.ndimage.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

def filter_3d_connected_components(volume):
    sz = None
    volume = volume.astype(np.int64)
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    # myarray = np.random.randint(1, 4, (11,11), dtype=np.int64)
    cc_volume = label(volume, connectivity=3)
    # print(f"this is ss {cc_volume}")
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume

def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=0.000001, mse=None):
    intersection = torch.sum(mse * real_mask, dim=[0, 1, 2, 3])
    union = torch.sum(mse, dim=[0, 1, 2, 3]) + torch.sum(real_mask, dim=[0, 1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    print(f"{intersection}, {union}")
    return dice, intersection, union

# lets code the function to plot the tsne from the anomaly dataset from different slices
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig('./latent_tsne.png')
    # plt.show()

def visualize_tsne_points_3d(tx, ty, tz, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter3D(current_tx, current_ty, current_tz, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()

# the function is to add coarse noise to tensor x
def add_coarse_noise(x, noise_std = 0.2, noise_res = 16):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    slice_size = x.shape[-1]
    ns = F.upsample_bilinear(ns, size=[slice_size, slice_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(slice_size))
    roll_y = random.choice(range(slice_size))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns

    return res

# the function is to add coarse noise to tensor x
def add_coarse_noise_lesser(x, noise_std = 0.1, noise_res = 16):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    slice_size = x.shape[-1]
    ns = F.upsample_bilinear(ns, size=[slice_size, slice_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(slice_size))
    roll_y = random.choice(range(slice_size))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns

    return res

def special_loss_disjoint_step1(y_true,y_pred):

    thresholded_pred = torch.where(y_pred < 0.0000000000000001, 1, y_pred)
    thresholded_true = torch.where(y_true < 0.0000000000000001, 1, y_true)

    return dice_coef_disloss(thresholded_true,thresholded_pred)

def special_loss_disjoint_step2(y_true,y_pred):

    thresholded_pred = torch.where(y_pred > 0.0000000000000001, 1, y_pred)
    thresholded_true = torch.where(y_true > 0.0000000000000001, 1, y_true)

    return dice_coef_disloss(thresholded_true,thresholded_pred)

def dice_coef_disloss(y_true, y_pred):

    y_true_f = y_true.view(y_true.shape[0], -1)
    y_pred_f = y_pred.view(y_pred.shape[0], -1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1)


def masked_mse(batch, batch_results):

        y = batch
        mask = batch.sum(dim=1, keepdim=True) > 0.01

        return (torch.pow(batch_results - y, 2) * mask.float()).mean()