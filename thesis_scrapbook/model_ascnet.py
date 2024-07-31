import torch
import os
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.modules.conv as conv


import argparse
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, input_channels = 1, dim=32, img_size = 256):
        super().__init__()
        self.dim = dim
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvBlock1_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2)
        )
        self.ConvBlock2_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4)
        )
        self.ConvBlock3_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
            nn.Conv2d(dim * 8, dim * 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8)
        )
        self.ConvBlock4_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 16),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 16)
        )

        self.fc = nn.Sequential(nn.Linear(16 * dim * int(img_size/16) ** 2, 1))
        self.tanh = nn.Tanh()

    def forward(self, img):
        out_1 = self.ConvBlock1(img)
        out_1 = self.ConvBlock1_Pool(out_1)

        out_2 = self.ConvBlock2(out_1)
        out_2 = self.ConvBlock2_Pool(out_2)

        out_3 = self.ConvBlock3(out_2)
        out_3 = self.ConvBlock3_Pool(out_3)

        out_4 = self.ConvBlock4(out_3)
        out_4 = self.ConvBlock4_Pool(out_4)

        out_5 = self.ConvBlock5(out_4)

        output = out_5.view(out_5.shape[0], -1)
        output = self.fc(output)
        output = self.tanh(output)

        return output

class Generator(nn.Module):
    def __init__(self, input_channels = 1, dim=32, img_size = 256):
        super().__init__()
        self.dim = dim

        #======================================================================encoder starts here
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvBlock1_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2)
        )
        self.ConvBlock2_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4)
        )
        self.ConvBlock3_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
            nn.Conv2d(dim * 8, dim * 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8)
        )
        self.ConvBlock4_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 16),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 16)
        )

        #======================================================================encoder ends here
        #======================================================================decoder1 starts here
        self.PreUpBlock6 =nn.Sequential(
            nn.ConvTranspose2d(16 * dim, 8 * self.dim, kernel_size=2, stride=2))
        self.UpBlock6 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(16 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
        )
        self.PreUpBlock7 =nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * self.dim, kernel_size=2, stride=2))
        self.UpBlock7 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(8 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
            nn.Conv2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
        )
        self.PreUpBlock8 =nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * self.dim, kernel_size=2, stride=2))
        self.UpBlock8 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(4 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
        )
        self.PreUpBlock9 =nn.Sequential(
            nn.ConvTranspose2d(2 * dim, self.dim, kernel_size=2, stride=2))
        self.UpBlock9 = nn.Sequential(
            nn.Conv2d(2 * self.dim, self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
        )

        self.LastConv = nn.Sequential(
            nn.Conv2d(self.dim, 1, kernel_size=1))
        self.Sigmoid = nn.Sigmoid()

        #======================================================================decoder1 ends here
        #======================================================================decoder2 starts here
        self.XPreUpBlock6 =nn.Sequential(
            nn.ConvTranspose2d(16 * dim, 8 * self.dim, kernel_size=2, stride=2))
        self.XUpBlock6 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(16 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
            nn.Conv2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 8),
        )

        self.XPreUpBlock7 =nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * self.dim, kernel_size=2, stride=2))
        self.XUpBlock7 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(8 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
            nn.Conv2d(4 * self.dim, 4 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 4),
        )

        self.XPreUpBlock8 =nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * self.dim, kernel_size=2, stride=2))
        self.XUpBlock8 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(4 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim * 2),
        )

        self.XPreUpBlock9 =nn.Sequential(
            nn.ConvTranspose2d(2 * dim, self.dim, kernel_size=2, stride=2))
        self.XUpBlock9 = nn.Sequential(
            nn.Conv2d(2 * self.dim, self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
        )

        self.XLastConv = nn.Sequential(
            nn.Conv2d(self.dim, 1, kernel_size=1))
        self.XSigmoid = nn.Sigmoid()

        #======================================================================decoder2 ends here
        #======================================================================combination module
        self.one_by_one = nn.Conv2d(2 , 1 , kernel_size= 1)
        self.combined_sigmoid = nn.Sigmoid()

    def forward(self, img):
        enc_1_pre = self.ConvBlock1(img)
        enc_1 = self.ConvBlock1_Pool(enc_1_pre)

        enc_2_pre = self.ConvBlock2(enc_1)
        enc_2 = self.ConvBlock2_Pool(enc_2_pre)

        enc_3_pre = self.ConvBlock3(enc_2)
        enc_3 = self.ConvBlock3_Pool(enc_3_pre)

        enc_4_pre = self.ConvBlock4(enc_3)
        enc_4 = self.ConvBlock4_Pool(enc_4_pre)

        enc_final = self.ConvBlock5(enc_4)
        #================================
        dec_1 = torch.cat((self.PreUpBlock6(enc_final), enc_4_pre), dim=1)
        dec_1 = self.UpBlock6(dec_1)
        dec_2 = torch.cat((self.PreUpBlock7(dec_1), enc_3_pre), dim=1)
        dec_2 = self.UpBlock7(dec_2)
        dec_3 = torch.cat((self.PreUpBlock8(dec_2), enc_2_pre), dim=1)
        dec_3 = self.UpBlock8(dec_3)
        dec_4 = torch.cat((self.PreUpBlock9(dec_3), enc_1_pre), dim=1)
        dec_4 = self.UpBlock9(dec_4)

        dec_final = self.LastConv(dec_4)
        dec_final = self.Sigmoid(dec_final)
        
        #================================
        xdec_1 = torch.cat((self.XPreUpBlock6(enc_final), enc_4_pre), dim=1)
        xdec_1 = self.XUpBlock6(xdec_1)
        xdec_2 = torch.cat((self.XPreUpBlock7(xdec_1), enc_3_pre), dim=1)
        xdec_2 = self.XUpBlock7(xdec_2)
        xdec_3 = torch.cat((self.XPreUpBlock8(xdec_2), enc_2_pre), dim=1)
        xdec_3 = self.XUpBlock8(xdec_3)
        xdec_4 = torch.cat((self.XPreUpBlock9(xdec_3), enc_1_pre), dim=1)
        xdec_4 = self.XUpBlock9(xdec_4)

        xdec_final = self.XLastConv(xdec_4)
        xdec_final = self.XSigmoid(xdec_final)

        # calculate the disjoincy loss here
        concat_final = torch.cat((dec_final, xdec_final), dim=1)

        # calculate the mse loss here
        final_gen = self.one_by_one(concat_final)
        final_gen = self.combined_sigmoid(final_gen)

        return dec_final, xdec_final, final_gen

if __name__ == "__main__":

    a = torch.randn(16, 1, 256, 256)

    disc = Discriminator()
    b = disc(a)
    print(b.shape)

    gen = Generator()
    c, d, e = gen(a)
    print(c.shape)
    print(d.shape)
    print(e.shape)