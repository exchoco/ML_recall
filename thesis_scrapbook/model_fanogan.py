import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""



class Generator_unified(nn.Module):
    def __init__(self, output_img_size = 256, intermediateResolutions = [8,8], outputChannels = 1, intermediateChannels = 16 , latent_dim = 128 , use_BatchNorm = False):
        super().__init__()

        # conv, reshape, dropout 0.1, dense
        self.intermediateChannels = intermediateChannels
        self.intermediateResolutions = intermediateResolutions
        self.fc = nn.Sequential(nn.Linear(int(latent_dim) , (intermediateChannels*(intermediateResolutions[0]**2))))
        self.DO1d = nn.Dropout1d(0.1)
        self.intermediate_conv = nn.Sequential(nn.Conv2d(intermediateChannels, intermediateChannels*8 , kernel_size=1, padding='same'))

        self.modules = []
        self.num_upsampling = int(math.log(output_img_size, 2) - math.log(float(intermediateResolutions[0]), 2))
        prev_filter = intermediateChannels*8
        if(use_BatchNorm):
            self.modules.append(nn.BatchNorm2d(prev_filter, 0.001, 0.99))
        else:
            self.modules.append(nn.LayerNorm([prev_filter, int(intermediateResolutions[0]), int(intermediateResolutions[1])]))
        self.modules.append(nn.ReLU())
        for i in range(self.num_upsampling):
            filters = int(max(32, 128 / (2 ** i)))
            self.modules.append(nn.ConvTranspose2d(prev_filter, filters, kernel_size=5, stride=2, padding=2, output_padding=1))
            if(use_BatchNorm):
                self.modules.append(nn.BatchNorm2d(filters, 0.001, 0.99))
            else:
                self.modules.append(nn.LayerNorm([filters, int(intermediateResolutions[0]*(2**(i+1))), int(intermediateResolutions[1]*(2**(i+1)))]))
            self.modules.append(nn.LeakyReLU())
            prev_filter = filters
        self.modules.append(nn.Conv2d(prev_filter, outputChannels, kernel_size=1, stride=1, padding="same"))
        self.seq = nn.Sequential(*self.modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.DO1d(out)
        out = out.view(-1, self.intermediateChannels, self.intermediateResolutions[0], self.intermediateResolutions[1])
        out = self.intermediate_conv(out)

        out =  self.seq(out)
        out = self.sigmoid(out)
        return out


class Discriminator_unified(nn.Module):
    def __init__(self, in_img_size = 256, intermediateResolutions=[8,8] , latent_dim=128 , use_BatchNorm = False):
        super().__init__()
        self.modules = []
        self.num_pooling = int(math.log(in_img_size, 2) - math.log(float(intermediateResolutions[0]), 2))
        prev_filter = 1
        for i in range(self.num_pooling):
          filters = int(min(128, 32 * (2 ** i)))
          self.modules.append(nn.Conv2d(prev_filter, filters, kernel_size=5, stride=2, padding=2))
          if(use_BatchNorm):
            self.modules.append(nn.BatchNorm2d(filters, 0.001, 0.99))
          else:
            self.modules.append(nn.LayerNorm([ int(in_img_size/(2**(i+1))), int(in_img_size/(2**(i+1)))]))
          self.modules.append(nn.LeakyReLU())
          prev_filter = filters
        self.seq = nn.Sequential(*self.modules)
        last_d = prev_filter * (in_img_size/(2**self.num_pooling))**2
        self.fc = nn.Sequential(nn.Linear(int(last_d) , 1))

    def forward(self, img):
        output = self.seq(img)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output

class Encoder_unified(nn.Module):
    def __init__(self, in_img_size = 256, intermediateResolutions=[8,8] , latent_dim=128 , use_BatchNorm = False):
        super().__init__()
        self.modules = []
        self.num_pooling = int(math.log(in_img_size, 2) - math.log(float(intermediateResolutions[0]), 2))
        prev_filter = 1
        for i in range(self.num_pooling):
          filters = int(min(128, 32 * (2 ** i)))
          self.modules.append(nn.Conv2d(prev_filter, filters, kernel_size=5, stride=2, padding=2))
          if(use_BatchNorm):
            self.modules.append(nn.BatchNorm2d(filters, 0.001, 0.99))
          else:
            self.modules.append(nn.LayerNorm([ int(in_img_size/(2**(i+1))), int(in_img_size/(2**(i+1)))]))
          self.modules.append(nn.LeakyReLU())
          prev_filter = filters
        self.seq = nn.Sequential(*self.modules)
        filters = prev_filter//8
        self.intermediate_conv = nn.Sequential(nn.Conv2d(prev_filter, filters, kernel_size=1, padding='same'))
        last_d = filters * (in_img_size/(2**self.num_pooling))**2
        self.fc = nn.Sequential(nn.Linear(int(last_d) , latent_dim))
        self.DO1d = nn.Dropout1d(0.1)
        self.tanh = nn.Tanh()

    def forward(self, img):
        output = self.seq(img)
        output = self.intermediate_conv(output)
        intermediate_feature_shape = output.shape[1]
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        output = self.DO1d(output)
        output = self.tanh(output)
        output = output.view(output.shape[0], -1)
        return output, intermediate_feature_shape

class Generator_128(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dim = opt.dim
        # self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,
                                          np.prod([opt.intermediateResolutions, opt.intermediateResolutions]) * 8 * self.dim))

        self.ResBlock1 = nn.Sequential(
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(8 * self.dim, 8 * self.dim, kernel_size=3, padding=1)
        )
        self.ResBlock2 = nn.Sequential(
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            # stridded conv transpose o = i * stride
            nn.ConvTranspose2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.ResBlock2_Shortcut = nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.ResBlock3 = nn.Sequential(
            nn.LayerNorm([2 * opt.intermediateResolutions, 2 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([2 * opt.intermediateResolutions, 2 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.ResBlock3_Shortcut = nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.ResBlock4 = nn.Sequential(
            nn.LayerNorm([4 * opt.intermediateResolutions, 4 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 1 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([4 * opt.intermediateResolutions, 4 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(1 * self.dim, 1 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.ResBlock4_Shortcut = nn.ConvTranspose2d(2 * self.dim, 1 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.layerNorm =  nn.LayerNorm([8 * opt.intermediateResolutions, 8 * opt.intermediateResolutions])
        self.conv2d = nn.Conv2d(1 * self.dim, 1, kernel_size=1,  padding='same')
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 8 * self.dim, self.opt.intermediateResolutions, self.opt.intermediateResolutions)
        out_1last = self.ResBlock1(out)
        out_1last = torch.add(out, out_1last)

        out_2 = self.ResBlock2(out_1last)
        out_2skip = self.ResBlock2_Shortcut(out_1last)
        out_2last = torch.add(out_2, out_2skip)

        out_3 = self.ResBlock3(out_2last)
        out_3skip = self.ResBlock3_Shortcut(out_2last)
        out_3last = torch.add(out_3, out_3skip)

        out_4 = self.ResBlock4(out_3last)
        out_4skip = self.ResBlock4_Shortcut(out_3last)
        out_4last = torch.add(out_4, out_4skip)

        output = self.layerNorm(out_4last)
        output = self.relu(output)
        output = self.conv2d(output)
        output = self.tanh(output)

        # print(f"block 1 output shape {out_1last.shape}")
        # print(f"block 2 output shape {out_2.shape}")
        # print(f"block 2 skip output shape {out_2skip.shape}")
        # print(f"block 2 last output shape {out_2last.shape}")
        # print(f"block 3 output shape {out_3.shape}")
        # print(f"block 3 skip output shape {out_3skip.shape}")
        # print(f"block 3 last output shape {out_3last.shape}")
        # print(f"block 4 output shape {out_4.shape}")
        # print(f"block 4 skip output shape {out_4skip.shape}")
        # print(f"block 4 last output shape {out_4last.shape}")
        # print(f"last relu conv tanh shape {output.shape}")
        # print(f"\n")
        return output

class Encoder(nn.Module):
    def __init__(self, opt, use_BatchNorm = True):
        super().__init__()
        self.modules = []
        self.num_pooling = int(math.log(opt.img_size, 2) - math.log(float(opt.intermediateResolutions), 2))
        prev_filter = 1
        for i in range(self.num_pooling):
          filters = int(min(128, 32 * (2 ** i)))
          self.modules.append(nn.Conv2d(prev_filter, filters, kernel_size=5, stride=2, padding=2))
          if(use_BatchNorm):
            self.modules.append(nn.BatchNorm2d(filters, 0.001, 0.99))
          else:
            self.modules.append(nn.LayerNorm([ int(opt.img_size/(2**(i+1))), int(opt.img_size/(2**(i+1)))]))
          self.modules.append(nn.LeakyReLU())
          prev_filter = filters
        self.seq = nn.Sequential(*self.modules)
        last_d = prev_filter * (opt.img_size/(2**self.num_pooling))**2
        self.fc = nn.Sequential(nn.Linear(int(last_d) , opt.latent_dim))
        self.tanh = nn.Tanh()

    def forward(self, img):
        output = self.seq(img)
        output = output.view(output.shape[0], -1)
        # print(f"output flat shape {output.shape}")
        output = self.fc(output)
        output=self.tanh(output)
        return output

class Encoder_UnList(nn.Module):
    def __init__(self, opt, use_BatchNorm = True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
  
        last_d = 128 * (opt.img_size/(2**4))**2
        self.fc = nn.Sequential(nn.Linear(int(last_d) , opt.latent_dim))
        self.tanh = nn.Tanh()

    def forward(self, img):
        output = self.block1(img)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = output.view(output.shape[0], -1)
        # print(f"output flat shape {output.shape}")
        output = self.fc(output)
        output=self.tanh(output)
        return output

class Discriminator_128(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dim = opt.dim
        self.dis_conv = nn.Conv2d(opt.channels, self.dim, kernel_size=3, stride=1, padding='same')
        self.ResBlock1 = nn.Sequential(
            nn.LayerNorm([opt.img_size, opt.img_size]),
            nn.ReLU(),
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.img_size, opt.img_size]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock1_Shortcut = nn.Sequential(
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock2 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/2), int(opt.img_size/2)]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/2), int(opt.img_size/2)]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock2_Shortcut = nn.Sequential(
            nn.Conv2d(2 * self.dim, 4 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock3 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/4), int(opt.img_size/4)]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/4), int(opt.img_size/4)]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock3_Shortcut = nn.Sequential(
            nn.Conv2d(4 * self.dim, 8 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock4 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/8), int(opt.img_size/8)]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/8), int(opt.img_size/8)]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same')
        )
        self.fc = nn.Sequential(nn.Linear(8 * self.dim * int(opt.img_size/8) ** 2, 1))

    def forward(self, img):
        output = self.dis_conv(img)
        out_1 = self.ResBlock1(output)
        out_1skip = self.ResBlock1_Shortcut(output)
        out_1last = torch.add(out_1, out_1skip)

        out_2 = self.ResBlock2(out_1last)
        out_2skip = self.ResBlock2_Shortcut(out_1last)
        out_2last = torch.add(out_2, out_2skip)

        out_3 = self.ResBlock3(out_2last)
        out_3skip = self.ResBlock3_Shortcut(out_2last)
        out_3last = torch.add(out_3, out_3skip)

        out_4 = self.ResBlock4(out_3last)
        out_4last = torch.add(out_4, out_3last)
        output = out_4last.view(out_4last.shape[0], -1)
        output = self.fc(output)
        # print(f"pre feature expansino block {output.shape}")
        # print(f"block 1 output shape {out_1.shape}")
        # print(f"block 1 skip output shape {out_1skip.shape}")
        # print(f"block 1 last output shape {out_1last.shape}")
        # print(f"block 2 output shape {out_2.shape}")
        # print(f"block 2 skip output shape {out_2skip.shape}")
        # print(f"block 2 last output shape {out_2last.shape}")
        # print(f"block 3 output shape {out_3.shape}")
        # print(f"block 3 skip output shape {out_3skip.shape}")
        # print(f"block 3 last output shape {out_3last.shape}")
        # print(f"block 4 output shape {out_4.shape}")
        # print(f"block 4 last output shape {out_4last.shape}")
        # print(f"last fully connected shape {output.shape}")
        # print(f"\n")
        return out_4last, output

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dim = opt.dim
        # self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,
                                          np.prod([opt.intermediateResolutions, opt.intermediateResolutions]) * 16 * self.dim))

        self.ResBlock0 = nn.Sequential(
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(16 * self.dim, 16 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(16 * self.dim, 16 * self.dim, kernel_size=3, padding=1)
        )

        self.ResBlock1 = nn.Sequential(
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(16 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.intermediateResolutions, opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.ResBlock1_Shortcut = nn.ConvTranspose2d(16 * self.dim, 8 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.ResBlock2 = nn.Sequential(
            nn.LayerNorm([2 * opt.intermediateResolutions, 2 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([2 * opt.intermediateResolutions, 2 * opt.intermediateResolutions]),
            nn.ReLU(),
            # stridded conv transpose o = i * stride
            nn.ConvTranspose2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.ResBlock2_Shortcut = nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.ResBlock3 = nn.Sequential(
            nn.LayerNorm([4 * opt.intermediateResolutions, 4 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([4 * opt.intermediateResolutions, 4 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.ResBlock3_Shortcut = nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.ResBlock4 = nn.Sequential(
            nn.LayerNorm([8 * opt.intermediateResolutions, 8 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 1 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([8 * opt.intermediateResolutions, 8 * opt.intermediateResolutions]),
            nn.ReLU(),
            nn.ConvTranspose2d(1 * self.dim, 1 * self.dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.ResBlock4_Shortcut = nn.ConvTranspose2d(2 * self.dim, 1 * self.dim, kernel_size=1, stride=2, output_padding=1)

        self.layerNorm =  nn.LayerNorm([16 * opt.intermediateResolutions, 16 * opt.intermediateResolutions])
        self.conv2d = nn.Conv2d(1 * self.dim, 1, kernel_size=1,  padding='same')
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 16 * self.dim, self.opt.intermediateResolutions, self.opt.intermediateResolutions)
        out_0last = self.ResBlock0(out)
        out_0last = torch.add(out, out_0last)

        out_1 = self.ResBlock1(out_0last)
        out_1skip = self.ResBlock1_Shortcut(out_0last)
        out_1last = torch.add(out_1, out_1skip)

        out_2 = self.ResBlock2(out_1last)
        out_2skip = self.ResBlock2_Shortcut(out_1last)
        out_2last = torch.add(out_2, out_2skip)

        out_3 = self.ResBlock3(out_2last)
        out_3skip = self.ResBlock3_Shortcut(out_2last)
        out_3last = torch.add(out_3, out_3skip)

        out_4 = self.ResBlock4(out_3last)
        out_4skip = self.ResBlock4_Shortcut(out_3last)
        out_4last = torch.add(out_4, out_4skip)

        output = self.layerNorm(out_4last)
        output = self.relu(output)
        output = self.conv2d(output)
        output = self.tanh(output)

        # print(f"block 1 output shape {out_1last.shape}")
        # print(f"block 2 output shape {out_2.shape}")
        # print(f"block 2 skip output shape {out_2skip.shape}")
        # print(f"block 2 last output shape {out_2last.shape}")
        # print(f"block 3 output shape {out_3.shape}")
        # print(f"block 3 skip output shape {out_3skip.shape}")
        # print(f"block 3 last output shape {out_3last.shape}")
        # print(f"block 4 output shape {out_4.shape}")
        # print(f"block 4 skip output shape {out_4skip.shape}")
        # print(f"block 4 last output shape {out_4last.shape}")
        # print(f"last relu conv tanh shape {output.shape}")
        # print(f"\n")
        return output


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dim = opt.dim
        self.dis_conv = nn.Conv2d(opt.channels, self.dim, kernel_size=3, stride=1, padding='same')
        self.ResBlock1 = nn.Sequential(
            nn.LayerNorm([opt.img_size, opt.img_size]),
            nn.ReLU(),
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([opt.img_size, opt.img_size]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock1_Shortcut = nn.Sequential(
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock2 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/2), int(opt.img_size/2)]),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/2), int(opt.img_size/2)]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock2_Shortcut = nn.Sequential(
            nn.Conv2d(2 * self.dim, 4 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock3 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/4), int(opt.img_size/4)]),
            nn.ReLU(),
            nn.Conv2d(4 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/4), int(opt.img_size/4)]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock3_Shortcut = nn.Sequential(
            nn.Conv2d(4 * self.dim, 8 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock4 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/8), int(opt.img_size/8)]),
            nn.ReLU(),
            nn.Conv2d(8 * self.dim, 16 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/8), int(opt.img_size/8)]),
            nn.ReLU(),
            nn.Conv2d(16 * self.dim, 16 * self.dim, kernel_size=3, stride=2, padding=1)
        )
        self.ResBlock4_Shortcut = nn.Sequential(
            nn.Conv2d(8 * self.dim, 16 * self.dim, kernel_size=1, stride=1, padding='same'),
            nn.AvgPool2d((2,2))
        )

        self.ResBlock5 = nn.Sequential(
            nn.LayerNorm([int(opt.img_size/16), int(opt.img_size/16)]),
            nn.ReLU(),
            nn.Conv2d(16 * self.dim, 16 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.LayerNorm([int(opt.img_size/16), int(opt.img_size/16)]),
            nn.ReLU(),
            nn.Conv2d(16 * self.dim, 16 * self.dim, kernel_size=3, stride=1, padding='same')
        )
        self.fc = nn.Sequential(nn.Linear(16 * self.dim * int(opt.img_size/16) ** 2, 1))

    def forward(self, img):
        output = self.dis_conv(img)
        out_1 = self.ResBlock1(output)
        out_1skip = self.ResBlock1_Shortcut(output)
        out_1last = torch.add(out_1, out_1skip)

        out_2 = self.ResBlock2(out_1last)
        out_2skip = self.ResBlock2_Shortcut(out_1last)
        out_2last = torch.add(out_2, out_2skip)

        out_3 = self.ResBlock3(out_2last)
        out_3skip = self.ResBlock3_Shortcut(out_2last)
        out_3last = torch.add(out_3, out_3skip)

        out_4 = self.ResBlock4(out_3last)
        out_4skip = self.ResBlock4_Shortcut(out_3last)
        out_4last = torch.add(out_4, out_4skip)

        out_5 = self.ResBlock5(out_4last)
        out_5last = torch.add(out_5, out_4last)
        output = out_5last.view(out_5last.shape[0], -1)
        output = self.fc(output)
        # print(f"pre feature expansino block {output.shape}")
        # print(f"block 1 output shape {out_1.shape}")
        # print(f"block 1 skip output shape {out_1skip.shape}")
        # print(f"block 1 last output shape {out_1last.shape}")
        # print(f"block 2 output shape {out_2.shape}")
        # print(f"block 2 skip output shape {out_2skip.shape}")
        # print(f"block 2 last output shape {out_2last.shape}")
        # print(f"block 3 output shape {out_3.shape}")
        # print(f"block 3 skip output shape {out_3skip.shape}")
        # print(f"block 3 last output shape {out_3last.shape}")
        # print(f"block 4 output shape {out_4.shape}")
        # print(f"block 4 last output shape {out_4last.shape}")
        # print(f"last fully connected shape {output.shape}")
        # print(f"\n")
        return out_4last, output


if __name__ == "__main__":
    gen = Generator_unified()
    enc = Encoder_unified()
    disc = Discriminator_unified()
    img = torch.randn(8, 1, 256, 256)
    b, fshape = enc(img)
    c = gen(b)
    d = disc(img)
    print(b.shape)
    print(c.shape)
    print(d.shape)

    class opt():
        def __init__(self):
            self.img_size = 256
            self.latent_dim = 128
            self.channels = 1
            self.dim = 64
            self.intermediateResolutions = 16

    opt = opt()
    a = torch.randn(8, 1, 256, 256)
    z = torch.randn(8, 128)
    model_G = Generator(opt)
    model_E = Encoder(opt)
    model_D = Discriminator(opt)
    out_G = model_G(z)
    out_E = model_E(a)
    outF_D, out_D = model_D(a)
    print(f"this is Gen output shape {out_G.shape}")
    print(f"this is Enc output shape {out_E.shape}")
    print(f"this is Disc output shape {out_D.shape}")
    print(f"this is Disc feature output shape {outF_D.shape}")