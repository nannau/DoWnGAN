import time
import numpy as np

from torch.nn.functional import avg_pool2d
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from gen_plots import write_to_board

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, device,
                 gp_weight=10, critic_iterations=5, print_every=10, save_every=5000,
                 use_cuda=True):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'Content': [], 'pca': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.save_every = save_every
        self.device = device

        assert next(self.G.parameters()).is_cuda
        assert next(self.D.parameters()).is_cuda

    def _critic_train_iteration(self, cr, hr):
        """ """
        # Get generated data
        batch_size = cr.size()[0]
        generated_data = self.G(cr)

        # Calculate probabilities on real and generated data
        data = Variable(hr, requires_grad=True)

        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

        del d_loss
        del gradient_penalty

    def _eof_loss(self, X, hr, fake):
        # Load PCA LHS
        real = torch.reshape(
            hr, 
            (hr.size(0), hr.size(1), hr.size(2)*hr.size(3))
        )

        fake = torch.reshape(
            fake, 
            (fake.size(0), fake.size(1), fake.size(2)*fake.size(3))
        )

        projected_real = torch.matmul(real, X.transpose(3, 2)).transpose(0, 1)
        projected_fake = torch.matmul(fake, X.transpose(3, 2)).transpose(0, 1)

        coefficient_loss = nn.L1Loss().to(self.device)
        closs = coefficient_loss(projected_fake, projected_real).item()
        return closs


    def _generator_train_iteration(self, cr, hr, pcas):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = cr.size()[0]
        generated_data = self.G(cr)

        X = pcas[:1, ...]
        eofloss = self._eof_loss(X, hr, generated_data)
        self.losses['pca'].append(eofloss)

        # get content loss
        content_loss = self._content_loss(cr, generated_data)
        self.losses['Content'].append(content_loss.item())

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()

        gamma = 0.01
        # g_loss = ((1-gamma/2)*eofloss + (1-gamma/2)*content_loss + gamma*g_loss)
        g_loss = (eofloss + content_loss + gamma*g_loss)

        g_loss.backward()
        self.G_opt.step()
        # Record loss
        self.losses['G'].append(g_loss.item())

        del g_loss
        del content_loss
        del eofloss

    def _content_loss(self, cr, generated_data):
        # Question is content loss compared to 
        # LR data or HR data? I've seen both.

        # Get generated data
        avg_pool = nn.AvgPool2d(4, stride=4)
        gen_lr = avg_pool(generated_data)

        criterion_pixelwise = nn.L1Loss().to(self.device)

        content_loss = criterion_pixelwise(cr, gen_lr)

        return content_loss


    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)#.to(self.device)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)#.to(self.device)

        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().detach().cpu())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, fixed, writer, epoch, timestr):
        for i, data in enumerate(data_loader):

            hr = data[0].clone().detach().requires_grad_(True).to(self.device)
            cr = data[1].clone().detach().requires_grad_(True).to(self.device)
            pcas = data[2].clone().detach().requires_grad_(True).to(self.device)

            self.num_steps += 1
            self._critic_train_iteration(cr, hr)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(cr, hr, pcas)

            if i % self.save_every == 0:
                # Save model progress
                torch.save(self.D.state_dict(), "trained_models/"+timestr+"_discriminator_model")
                torch.save(self.G.state_dict(), "trained_models/"+timestr+"_generator_model")

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                print(f"cuda memory: {torch.cuda.memory_allocated()}")

                if self.num_steps > self.critic_iterations:
                    loss_dict = {
                        'D': self.losses['D'][-1],
                        'GP': self.losses['GP'][-1],
                        'Gradient Norm': self.losses['gradient_norm'][-1],
                        'G': self.losses['G'][-1],
                        'Content': self.losses['Content'][-1],
                        'pca': self.losses['pca'][-1],
                    }

                    loss_dict_writer = {
                        'D': self.losses['D'],
                        'GP': self.losses['GP'],
                        'Gradient Norm': self.losses['gradient_norm'],
                        'G': self.losses['G'],
                        'Content': self.losses['Content'],
                        'pca': self.losses['pca'],
                    }

                    fixed_writer = {}
                    fixed_writer["fake"] = self.G(fixed["coarse"]).detach().cpu()
                    fixed_writer["real"] = fixed["fine"].detach().cpu()
                    fixed_writer["coarse"] = fixed["coarse"].detach().cpu()

                    write_to_board(writer, loss_dict, loss_dict_writer, fixed_writer, epoch*len(data_loader)+i)

                    print("G: {}".format(self.losses['G'][-1]))
                    print("Content loss: {}".format(self.losses['Content'][-1]))
                    print("EOF: ", self.losses['pca'][-1])

                self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'Content': [], 'pca': []}


    def train(self, data_loader, epochs, fixed, save_training_gif=True):

        timestr = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter()

        # Fix latents to see how image generation improves during training
        fixed_coarse = Variable(fixed["coarse"], requires_grad=True)
        fixed_real = Variable(fixed["fine"], requires_grad=True)

        if self.use_cuda:
            fixed_coarse = fixed_coarse.to(self.device)
            fixed_real = fixed_real.to(self.device)

        training_progress_images = []

        fixed_writer = {
            "coarse": fixed_coarse,
            "fine": fixed_real
        }

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader, fixed_writer, writer, epoch, timestr)

