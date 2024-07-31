import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.autograd as autograd
from torchvision.utils import save_image
import datetime
# from fanogan.train_wgangp import train_wgangp
from dataset import SkullStrippedinit_datasets, init_dataset_loader, init_dataset_loader_nocycle, AnomalousMRIDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    a, b = D(interpolates)
    d_interpolates = b
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgangp(opt, generator, discriminator, dataloader, device):
    t_log_dir = "./WGAN_Log/"
    log_writer = SummaryWriter(os.path.join(t_log_dir,str(datetime.datetime.now())))
    generator.to(device)
    discriminator.to(device)
    scale = 10.0
    kappa = 1.0
    
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("./weight", exist_ok=True)
    os.makedirs("./mod_weight", exist_ok=True)
    os.makedirs("./train_data_out/generated_fakes/", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))
    
    batches_done = 0
    for epoch in range(opt.n_epochs):
        generator.train()
        discriminator.train()
        batch_GLoss = []
        batch_DLoss = []
        for i, (x)in enumerate(dataloader):
            imgs = x["anchor"]

            real_imgs = imgs.to(device)
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            fake_imgs = generator(z)
            # Real images
            fake_features, fake_validity = discriminator(fake_imgs)
            # here the self['d_'] is fake_validity as we passed fake generated from z to disc
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(opt.n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
                # Generate a batch of images
                fake_imgs = generator(z)
                # Real images
                real_features, real_validity = discriminator(real_imgs)
                # Fake images
                fake_features, fake_validity = discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator,
                                                            real_imgs.data,
                                                            fake_imgs.data,
                                                            device)
                # Adversarial loss
                d_loss = (torch.mean(fake_validity) - torch.mean(real_validity)
                        + scale * gradient_penalty)
                d_loss.backward()
                optimizer_D.step()

            print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                    f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():3f}] "
                    f"[G loss: {g_loss.item():3f}]")

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25],
                            f"./train_data_out/generated_fakes/{batches_done:06}.png",
                            nrow=5, normalize=True)
            batch_GLoss.append(g_loss.item())
            batch_DLoss.append(d_loss.item())
            batches_done += opt.n_critic
        log_writer.add_scalar('generator_loss', np.average(np.array(batch_GLoss)), epoch)
        log_writer.add_scalar('discriminator_loss', np.average(np.array(batch_DLoss)), epoch)
        if(epoch % 10 ==0):
                print("saving weight....") 
                weight_path = './weight/Generator%d.pt' %(int(epoch))
                torch.save(generator.module.state_dict(), weight_path)
                weight_path = './weight/Discriminator%d.pt' %(int(epoch))
                torch.save(discriminator.module.state_dict(), weight_path)

                weight_path = './mod_weight/Generator%d.pt' %(int(epoch))
                torch.save(generator.state_dict(), weight_path)
                weight_path = './mod_weight/Discriminator%d.pt' %(int(epoch))
                torch.save(discriminator.state_dict(), weight_path)
        torch.cuda.empty_cache()
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
    # testing_dataset_loader = torch.utils.data.DataLoader(
    #                 testing_dataset,
    #                 batch_size= 8, shuffle=True,
    #                 num_workers=0, drop_last=True
    #                 )

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from model_fanogan import Generator, Discriminator, Generator_128, Discriminator_128
    generator = Generator_128(opt)
    discriminator = Discriminator_128(opt)
    generator = nn.DataParallel(generator, device_ids=[0,1,3])
    discriminator = nn.DataParallel(discriminator, device_ids=[0,1,3])
    train_wgangp(opt, generator, discriminator, training_dataset_loader, device)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=12,
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

    parser.add_argument("--sample_interval", type=int, default=150,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
