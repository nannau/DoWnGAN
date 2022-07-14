from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.losses import content_loss#, kinetic_energy_loss
from DoWnGAN.mlflow_tools.gen_grid_plots import gen_grid_images
from DoWnGAN.mlflow_tools.mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models

import torch
from torch.autograd import grad as torch_grad

import mlflow

torch.autograd.set_detect_anomaly(True)


class WassersteinGAN:
    """Implements Wasserstein GAN with gradient penalty and 
    frequency separation"""

    def __init__(self, G, C, G_optimizer, C_optimizer) -> None:
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.num_steps = 0
        
    def _critic_train_iteration(self, coarse, fine):
        """
        Performs one iteration of the critic training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """

        fake = self.G(coarse)

        c_real = self.C(fine)
        c_fake = self.C(fake)

        gradient_penalty = hp.gp_lambda * self._gp(fine, fake, self.C)

        # Zero the gradients
        self.C_optimizer.zero_grad()


        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        w_estimate = c_real_mean - c_fake_mean

        critic_loss.backward(retain_graph = True)

        # Update the critic
        self.C_optimizer.step()


    def _generator_train_iteration(self, coarse, fine):
        """
        Performs one iteration of the generator training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        self.G_optimizer.zero_grad()

        fake = self.G(coarse)
        c_fake = self.C(fake)

        # EK = kinetic_energy_loss(fine, fake)

        # Calculate generator loss
        # Uncomment for KE loss only
        g_loss = -torch.mean(c_fake)*hp.gamma
        # g_loss = g_loss + 10*EK

        # Add content loss and create objective function
        g_loss += hp.content_lambda * content_loss(fake, fine, device=config.device)

        g_loss.backward()

        # Update the generator
        self.G_optimizer.step()



    def _gp(self, real, fake, critic):
        current_batch_size = real.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=config.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data

        # Calculate probability of interpolated examples
        critic_interpolated = critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=config.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(hp.batch_size, -1).to(config.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return hp.gp_lambda * ((gradients_norm - 1) ** 2).mean()


    def _train_epoch(self, dataloader, testdataloader, epoch):
        """
        Performs one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            epoch (int): The epoch number.
        """
        print(80*"=")
        train_metrics = initialize_metric_dicts({})
        test_metrics = initialize_metric_dicts({})

        for data in dataloader:
            coarse = data[0].to(config.device)
            fine = data[1].to(config.device)
            self._critic_train_iteration(coarse, fine)

            if self.num_steps%hp.critic_iterations == 0:
                self._generator_train_iteration(coarse, fine)

            # Track train set metrics
            train_metrics = gen_batch_and_log_metrics(
                self.G,
                self.C,
                coarse,
                fine,
                train_metrics,
            )
            self.num_steps += 1

        # Take mean of all batches and log to file
        with torch.no_grad():
            post_epoch_metric_mean(train_metrics, "train")

            # Generate plots from training set
            cbatch, rbatch = next(iter(dataloader))
            gen_grid_images(self.G, cbatch, rbatch, epoch, "train")

            test_metrics = initialize_metric_dicts({})
            for data in testdataloader:
                coarse = data[0].to(config.device)
                fine = data[1].to(config.device)

                # Track train set metrics
                test_metrics = gen_batch_and_log_metrics(
                    self.G,
                    self.C,
                    coarse,
                    fine,
                    test_metrics,
                )

            # Take mean of all batches and log to file
            post_epoch_metric_mean(test_metrics, "test")

            cbatch, rbatch = next(iter(testdataloader))
            gen_grid_images(self.G, cbatch, rbatch, epoch, "test")

            # Log the models to mlflow pytorch models
            print(f"Artifact URI: {mlflow.get_artifact_uri()}")
            log_network_models(self.C, self.G, epoch)

    def train(self, dataloader, testdataloader):
        """
        Trains the model.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
        """
        self.num_steps = 0
        for epoch in range(hp.epochs):
            self._train_epoch(dataloader, testdataloader, epoch)
