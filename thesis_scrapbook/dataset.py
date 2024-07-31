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
from torchvision import transforms
from imageio import imread
import pandas as pd
import argparse
from numpy import append
import re
from simplex import Simplex_CLASS
from utils import *
from random import randint
from scipy.ndimage.interpolation import rotate
from utils import *
import sys
import cv2
np.set_printoptions(threshold=sys.maxsize)

class SkullStrippedMRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(256, 256), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                #  transforms.RandomAffine(3, translate=(0.02, 0.09)),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 transforms.ToTensor()
                 ]
                ) if not transform else transform
        self.filenames = []
        for filename in os.scandir(ROOT_DIR):
            if (os.path.isdir(filename.path) and filename.name.endswith("-T2")):
                self.filenames.append(filename.name)
        print(f"this is file list {self.filenames} and length {len(self.filenames)}")
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice
        
        self.simplex = Simplex_CLASS()
        betas = get_beta_schedule(1000, "linear")
        alphas = 1 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.img_size = img_size

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_list = [idx, np.random.randint(0, len(self.filenames)), np.random.randint(0, len(self.filenames))]
        # idx_list = [idx, 5, 5]
        # print(f"this is the random index {idx_list}")
        images = []
        slice_list = []
        for x, idx in enumerate(idx_list):
            # wait
            # if(1==2):
            #     exit()
            if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
                image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
            else:
                img_name = os.path.join(
                        self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.nii"
                        )
                img = nib.load(img_name)

                # # wait
                # print(img_name)
                # print(img.header)
                # sx, sy, sz = img.header.get_zooms()
                # print(sx, sy, sz)


                image = img.get_fdata()

                image_mean = np.mean(image)
                image_std = np.std(image)
                img_range = (image_mean - 4 * image_std, image_mean + 6 * image_std)
                image = np.clip(image, img_range[0], img_range[1])
                image = image / (img_range[1] - img_range[0])
                np.save(
                        os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                                np.float32
                                )
                        )
            
            # get the mask
            if(x == 0):
                mask_name = os.path.join(
                        self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_mask.nii"
                        )
                msk = nib.load(mask_name)
                image1_mask = msk.get_fdata()
            elif(x == 1):
                mask_name = os.path.join(
                        self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_mask.nii"
                        )
                msk = nib.load(mask_name)
                image2_mask = msk.get_fdata()
            elif(x == 2):
                mask_name = os.path.join(
                        self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_mask.nii"
                        )
                msk = nib.load(mask_name)
                image3_mask = msk.get_fdata()

            if self.random_slice:
                slice_idx = randint(0, image.shape[2]-1)
                while(np.sum(image[:,:, slice_idx:slice_idx + 1].flatten()) <= 1):
                    slice_idx = randint(0, image.shape[2]-1)
            else:
                slice_idx = 58
            
            slice_list.append(slice_idx)

            image = image[:,:, slice_idx:slice_idx + 1].reshape(self.img_size[0], self.img_size[1]).astype(np.float32)
            image = np.rot90(image, 3)
            images.append(image)

            if(x == 0):
                image1_mask = image1_mask[:,:, slice_idx:slice_idx + 1].reshape(self.img_size[0], self.img_size[1]).astype(np.float32)
                image1_mask = np.rot90(image1_mask, 3)
            elif(x == 1):
                image2_mask = image2_mask[:,:, slice_idx:slice_idx + 1].reshape(self.img_size[0], self.img_size[1]).astype(np.float32)
                image2_mask = np.rot90(image2_mask, 3)
            elif(x == 2):
                image3_mask = image3_mask[:,:, slice_idx:slice_idx + 1].reshape(self.img_size[0], self.img_size[1]).astype(np.float32)
                image3_mask = np.rot90(image3_mask, 3)
        
        # print(f"this is the random slice index {slice_list}")
        if self.transform:
            image1 = self.transform(images[0])
            image2 = self.transform(images[1])
            image3 = self.transform(images[2])
            image1_mask = self.transform(image1_mask)
            image2_mask = self.transform(image2_mask)
            image3_mask = self.transform(image3_mask)
        # print(f"this is image tensor shape {image3.shape}")

        reshaped_image3 = image3.reshape(1, image3.shape[0], image3.shape[1], image3.shape[2])
        t_tensor = torch.tensor([88 - 1], device=reshaped_image3.device).repeat(reshaped_image3.shape[0])
        # t_tensor = torch.tensor([torch.randint(50 -1, 100 -1, (1,))], device=reshaped_image3.device).repeat(reshaped_image3.shape[0])
        # print(f"this is the timestep T {t_tensor}")
        noise = generate_simplex_noise(self.simplex, reshaped_image3, t_tensor, False, in_channels=1)
        noisy_image3 = self.sample_q(
                    reshaped_image3, t_tensor,
                    noise.float()
                    )
        noisy_image3 = noisy_image3.view(reshaped_image3.shape[1], reshaped_image3.shape[2], reshaped_image3.shape[3])
        noisy_image3 = noisy_image3 * image3_mask


        # anchor_selector = np.random.randint(0, 2)
        anchor_selector = 1
        if(anchor_selector == 1):
            anchor = image1
            positive = image2
            negative = noisy_image3
            negative_ori = image3

            # add noised anchor for batch random control to make sure slices from each subject appear at least once in each epoch
            if self.transform:
                image1_mask = self.transform(image1_mask)

            reshaped_image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])
            t_tensor = torch.tensor([88 - 1], device=reshaped_image3.device).repeat(reshaped_image3.shape[0])
            noise = generate_simplex_noise(self.simplex, reshaped_image1, t_tensor, False, in_channels=1)
            noisy_image1 = self.sample_q(
                        reshaped_image1, t_tensor,
                        noise.float()
                        )
            noisy_image1 = noisy_image1.view(reshaped_image1.shape[1], reshaped_image1.shape[2], reshaped_image1.shape[3])
            noisy_image1 = noisy_image1 * image1_mask
        else:
            if self.transform:
                image1_mask = self.transform(image1_mask)

            reshaped_image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])
            t_tensor = torch.tensor([88 - 1], device=reshaped_image3.device).repeat(reshaped_image3.shape[0])
            noise = generate_simplex_noise(self.simplex, reshaped_image1, t_tensor, False, in_channels=1)
            noisy_image1 = self.sample_q(
                        reshaped_image1, t_tensor,
                        noise.float()
                        )
            noisy_image1 = noisy_image1.view(reshaped_image1.shape[1], reshaped_image1.shape[2], reshaped_image1.shape[3])
            noisy_image1 = noisy_image1 * image1_mask

            anchor = noisy_image1
            positive = noisy_image3
            negative = image2
        
        # # test the coarse noise
        # coarsed = add_coarse_noise(anchor[None, : , : , :])
        # print(f"coarsed shape {coarsed.shape}")
        # # print(f"the selector for anchor is {anchor_selector}")
        # plt.imshow(coarsed[0][0], cmap="gray")
        # plt.savefig("./test_coarse.png")
        # # add noise to the anchor only for comparison
        # reshaped_image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])
        # noisy_image1 = self.sample_q(
        #             reshaped_image1, t_tensor,
        #             noise.float()
        #             )
        # noisy_image1 = noisy_image1.view(reshaped_image3.shape[1], reshaped_image3.shape[2], reshaped_image3.shape[3])
        # noisy_image1 = noisy_image1 * image1_mask

        # noisy_image1 = torch.where(noisy_image1 > 0.0000000001, noisy_image1, 0)
        # plt.imshow((noisy_image1 * image1_mask)[0], cmap="gray")
        # plt.savefig("./test_simplex.png")
        # plt.imshow((anchor * image1_mask.int())[0], cmap="gray")
        # plt.savefig("./test_ori.png")
        # print(anchor.amax())
        # print(anchor.amin())
        # coarsed_image = add_coarse_noise(anchor[None, :, :, :].clone())
        # coarsed_image = torch.where(coarsed_image > 0.0000000001, coarsed_image, 0)
        # plt.imshow((coarsed_image[0] * image1_mask)[0], cmap="gray")
        # plt.savefig("./test_coarse.png")

        # show_single_image(anchor, positive)
        sample = {'anchor': anchor, "noisy_anchor":noisy_image1, "positive": positive, "negative": negative, "negative_ori": negative_ori, "anchor_mask": image1_mask, "positive_mask": image2_mask, "negative_mask": image3_mask}
        return sample

def SkullStrippedinit_datasets(args_img_size=(256, 256), args_random_slice=True):
    training_dataset = SkullStrippedMRIDataset(
            ROOT_DIR="../Transcend/IXI_T2_small_ss", img_size=args_img_size, random_slice=args_random_slice
            )
    testing_dataset = SkullStrippedMRIDataset(
            ROOT_DIR="../Transcend/IXI_T2_small_valid_ss", img_size=args_img_size, random_slice=args_random_slice
            )
    return training_dataset, testing_dataset

class AnomalousMRIDataset(Dataset):
    """Anomalous MRI dataset."""

    def __init__(
            self, ROOT_DIR, transform=None, img_size=(256, 256), slice_selection="iterateKnown_restricted", resized=False,
            cleaned=True
            ):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_size: size of each 2D dataset image
            slice_selection: "random" = randomly selects a slice from the image
                             "iterateKnown" = iterates between ranges of tumour using slice data
                             "iterateUnKnown" = iterates through whole MRI volume
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                #  transforms.CenterCrop((175, 240)),
                 # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor()
                #  transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform
        self.img_size = img_size
        self.resized = resized
        self.slices = {
            "001": range(37, 107), "002": range(35, 79), "003": range(64, 110), "004": range(57, 119),
            "005": range(82, 130), "006": range(61, 134), "007": range(37, 86), "008": range(26, 71),
            "009": range(33, 109), "010": range(70, 123), "011": range(47, 112), "012": range(49, 83),
            "013": range(71, 112), "014": range(29, 87), "015": range(40, 113), "016": range(58, 133),
            "017": range(46, 114), "018": range(51, 101), "019": range(33, 80), "020": range(18, 92),
            "021": range(90, 129), "022": range(72, 129), "023": range(77, 124), "024": range(36, 118),
            "025": range(78, 134), "026": range(36, 108), "027": range(39, 102), "028": range(38, 68),
            "029": range(49, 109), "030": range(51, 122)
            }
        self.filenames = self.slices.keys()
        if cleaned:
            self.filenames = list(map(lambda name: f"{ROOT_DIR}/raw_cleaned/{name}.npy", self.filenames))
        else:
            self.filenames = list(map(lambda name: f"{ROOT_DIR}/raw/{name}.npy", self.filenames))
        # changed here
        self.filenames = sorted(os.listdir(ROOT_DIR+"/raw_cleaned"))

        # remove duplicate
        for idx, x in enumerate(self.filenames):
            self.filenames[idx] = self.filenames[idx][:-4]
        self.filenames = sorted(list(set(self.filenames)))
        for idx, x in enumerate(self.filenames):
            self.filenames[idx] = self.filenames[idx]+".nii"
        print(f"the filenames list : {self.filenames}")

        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.slice_selection = slice_selection
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        labels = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # # wait
        # if(1==2):
        #     exit()

        if os.path.exists(os.path.join(self.ROOT_DIR+"/raw_cleaned/"+f"{self.filenames[idx][:-4]}.npy")):
            if self.resized and os.path.exists(os.path.join(f"{self.filenames[idx][:-4]}-resized.npy")):
                image = np.load(os.path.join(f"{self.filenames[idx][:-4]}-resized.npy"))
            else:
                image = np.load(os.path.join(self.ROOT_DIR+"/raw_cleaned/"+f"{self.filenames[idx][:-4]}.npy"))
            # print(f"this is image file name {self.filenames[idx][:-4]}")
        else:
            # img_name = os.path.join(self.filenames[idx])

            # changed here
            img_name = os.path.join(self.ROOT_DIR+"/raw_cleaned", self.filenames[idx])
            # print(f"this is image file name {img_name}")
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()
            image = np.rot90(image)

            # # wait
            # print(img_name)
            # print(img.header)
            # sx, sy, sz = img.header.get_zooms()
            # print(sx, sy, sz)


            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 4 * image_std, image_mean + 6 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(self.ROOT_DIR+"/raw_cleaned/"+f"{self.filenames[idx][:-4]}.npy"), image.astype(
                            np.float32
                            )
                    )
        # print(f"this is image npy shape {image.shape}")
        sample = {}

        if self.resized:
            img_mask = np.load(f"{self.ROOT_DIR}/mask/{self.filenames[idx][-9:-4]}-resized.npy")
        else:
            # img_mask = np.load(f"{self.ROOT_DIR}/mask/{self.filenames[idx][-9:-4]}.npy")
            # changed here
            msk_name = os.path.join(self.ROOT_DIR+"/mask", self.filenames[idx][:-6]+"seg.nii")
            # print(f"this is mask file name {msk_name}")
            msk = nib.load(msk_name)
            mask = msk.get_fdata()
            mask = np.rot90(mask)
            img_mask = mask


        if self.slice_selection == "iterateKnown_restricted":

            # temp_range = self.slices[self.filenames[idx][-9:-4]]
            # changed here

            temp_range = range(0, image.shape[2]-1)
            # temp_range = self.slices[self.filenames[idx][-10:-7]]

            # output = torch.empty(4, *self.img_size)
            # output_mask = torch.empty(4, *self.img_size)
            # changed here 

            output = torch.empty( temp_range.stop-temp_range.start, *self.img_size)
            output_mask = torch.empty( temp_range.stop-temp_range.start, *self.img_size)

            # slices = np.linspace(temp_range.start + 5, temp_range.stop - 5, 4).astype(np.int32)
            # print(f"using the slices {slices}")
            # changed here
            slices = np.linspace(temp_range.start, temp_range.stop, temp_range.stop-temp_range.start ).astype(np.int32)

            for counter, i in enumerate(slices):
                # temp = image[i, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                # temp_mask = img_mask[i, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                # changed here
                temp = image[:,:, i].reshape(image.shape[0], image.shape[1]).astype(np.float32)
                temp_mask = img_mask[:,:, i].reshape(image.shape[0], image.shape[1]).astype(np.float32)

                # wait
                if self.transform:
                    temp = self.transform(temp)
                    temp_mask = self.transform(temp_mask)
                output[counter, ...] = temp
                output_mask[counter, ...] = temp_mask

                # create labels list
                if(i in self.slices[self.filenames[idx][-10:-7]]):
                    labels.append(1)
                else:
                    labels.append(0)

            image = output
            sample["slices"] = slices
            sample["mask"] = (output_mask.view(-1, 1, output_mask.shape[-2], output_mask.shape[-1]) > 0).float()

        sample["image"] = image.view(-1, 1, image.shape[-2], image.shape[-1])
        sample["filenames"] = self.filenames[idx]
        # show_single_image(image[50].reshape(-1,256,256), sample["mask"][50].reshape(-1,256,256))

        # # show image for checking
        # print(f"this is image npy shape after transform {image.shape}")
        # mask_im = sample["mask"]
        # print(f"this is mask npy shape after transform {mask_im.shape}")
        # image_im = image[50]
        
        # image_arr = np.array(image_im.cpu(), dtype=np.uint8)
        # # image_arr = np.array(sample["mask"][50], dtype=np.uint8)
        # fig = plt.figure(figsize=(8, 8))
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(  image_arr, cmap="gray" )
        # plt.savefig("./test_anomaly.png")
        # # plt.show()

        # add label for the tsne plot 0 for normal slice and 1 for anomaly slice
        sample["label"] = np.array(labels)
        return sample

def init_dataset_loader(mri_dataset,args_batch_size, args_shuffle=True):
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=args_batch_size, shuffle=args_shuffle,
                    num_workers=0, drop_last=True
                    )
            )

    return dataset_loader

def init_dataset_loader_nocycle(mri_dataset,args_batch_size, args_shuffle=True):
    dataset_loader = torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=args_batch_size, shuffle=args_shuffle,
                    num_workers=0, drop_last=True
                    )

    return dataset_loader

if __name__ == "__main__":

    # d_set = SkullStrippedMRIDataset(
    #         ROOT_DIR="../Transcend/IXI_T2_small_ss", img_size=[256,256], random_slice=False
    #         )
    # training_dataset_loader = torch.utils.data.DataLoader(
    #                 d_set,
    #                 batch_size=32, shuffle=True,
    #                 num_workers=0, drop_last=True
    #                 )
    # train_img_shape = d_set.__getitem__(146)["negative"].shape
    # print(f"this is the train dataset image shape {train_img_shape}")

    a_set = AnomalousMRIDataset(
            ROOT_DIR='../23_01_15_AnoDDPM/AnoDDPM/DATASETS/CancerousDataset/BraTS_T2'
            )
    eval_dataset_loader = torch.utils.data.DataLoader(
                    a_set,
                    batch_size= 8, shuffle=False,
                    num_workers=0, drop_last=True
                    )

    eval_img_shape = a_set.__getitem__(1)["mask"].shape
    print(f"this is the eval dataset image shape {eval_img_shape}")

    # for x in eval_dataset_loader:
    #     # create_folder_save_tensor_imgs("train-out", 32, "hehe.png", x['anchor'], x['positive'])
    #     break