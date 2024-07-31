import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import datetime
# from fanogan.train_wgangp import train_wgangp
from dataset import SkullStrippedinit_datasets, init_dataset_loader, init_dataset_loader_nocycle, AnomalousMRIDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import *

def train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, test_dataloader, device, kappa=1.0):
    t_log_dir = "./Enc_Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))
    generator.load_state_dict(torch.load("./mod_weight/Generator260.pt"))
    discriminator.load_state_dict(torch.load("./mod_weight/Discriminator260.pt"))
    # encoder.load_state_dict(torch.load("./results/Encoder5.pt"))
    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(train_dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        # generator.train()
        # discriminator.train()
        encoder.train()
        batch_ELoss = []
        for i, (x) in enumerate(train_dataloader):
            imgs = x["anchor"]
            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features, real_validity  = discriminator.forward(real_imgs)
            # Fake features
            fake_features, fake_validity = discriminator.forward(fake_imgs)

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()
        batches_done += len(train_dataloader)
        # print epoch last mini batch lose
        print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                f"[E loss: {e_loss.item():3f}]")

        if epoch % opt.sample_interval == 0:
            print("saving train samples")
            # fake_z = encoder(fake_imgs)
            # reconfiguration_imgs = generator(fake_z)
            create_folder_save_tensor_imgs("train_data_out/anchor", opt.batch_size // 4, f"train_data_epoch{epoch}.png", real_imgs[-opt.batch_size // 4:, ...] , fake_imgs[-opt.batch_size // 4:, ...])

        batch_ELoss.append(e_loss.item())
        log_writer.add_scalar('encoder_loss', np.average(np.array(batch_ELoss)), epoch)

        if(epoch % 10 == 0 ):
            with torch.no_grad():
                encoder.eval()
                for i, (x) in enumerate(test_dataloader):
                    imgs = x["anchor"]
                    # Configure input
                    real_imgs = imgs.to(device)

                    # Generate a batch of latent variables
                    z = encoder(real_imgs)

                    # Generate a batch of images
                    fake_imgs = generator(z)

                    if i == 0:
                        print("saving eval samples")
                        # fake_z = encoder(fake_imgs)
                        # reconfiguration_imgs = generator(fake_z)
                        create_folder_save_tensor_imgs("eval_data_out/anchor", opt.batch_size // 4, f"eval_data_epoch{epoch}.png", real_imgs[-opt.batch_size // 4:, ...] , fake_imgs[-opt.batch_size // 4:, ...])

        if(epoch % 10 == 0):
            weight_path = './weight/Encoder%d.pt' %(int(epoch))
            torch.save(encoder.module.state_dict(), weight_path)

            weight_path = './mod_weight/Encoder%d.pt' %(int(epoch))
            torch.save(encoder.state_dict(), weight_path)

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_dataset, testing_dataset = SkullStrippedinit_datasets()
    # training_dataset_loader = init_dataset_loader(training_dataset, args.batch_size)
    # testing_dataset_loader = init_dataset_loader(testing_dataset, args.batch_size)
    training_dataset_loader = torch.utils.data.DataLoader(
                    training_dataset,
                    batch_size=opt.batch_size, shuffle=True,
                    num_workers=0, drop_last=True
                    )
    testing_dataset_loader = torch.utils.data.DataLoader(
                    testing_dataset,
                    batch_size= 8, shuffle=True,
                    num_workers=0, drop_last=True
                    )

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from model_fanogan import Generator, Discriminator, Encoder, Encoder_UnList, Generator_128, Discriminator_128

    generator = Generator_128(opt)
    discriminator = Discriminator_128(opt)
    encoder = Encoder_UnList(opt)
    generator = nn.DataParallel(generator, device_ids=[0,1,3])
    discriminator = nn.DataParallel(discriminator, device_ids=[0,1,3])
    encoder = nn.DataParallel(encoder, device_ids=[0,1,3])

    train_encoder_izif(opt, generator, discriminator, encoder,
                       training_dataset_loader, testing_dataset_loader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--dim", type=int, default=64,
                        help="number of feature dimension")
    parser.add_argument("--intermediateResolutions", type=int, default=32,
                        help="number of intermediate feature size")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for discriminator per iter")

    parser.add_argument("--sample_interval", type=int, default=5,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
