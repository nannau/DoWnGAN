# This defines the wasserstein architecture
from DoWnGAN.gen_grid_plots import gen_grid_images
from torch.autograd import grad as torch_grad

import stage as s
import hyperparams as hp
from DoWnGAN.losses import content_loss
from DoWnGAN.gen_grid_plots import gen_grid_images
import mlflow
from mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models
import torch, gc

# torch.autograd.set_detect_anomaly(True)

class WassersteinGANFS:
    """Implements Wasserstein GAN with gradient penalty and 
    frequency separation"""

    def __init__(self, G, C, G_optimizer, C_optimizer) -> None:
        self.G = s.generator
        self.C = s.critic
        self.G_optimizer = s.G_optimizer
        self.C_optimizer = s.C_optimizer
        self.num_steps = 0
        gc.collect()
        torch.cuda.empty_cache()

    def _critic_train_iteration(self, coarse, fine):
        """
        Performs one iteration of the critic training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """

        fake = self.G(coarse)
        fake_low = hp.low(hp.rf(fake))
        real_low = hp.low(hp.rf(fine))

        fake_high = fake - fake_low
        real_high = fine - real_low

        c_real = self.C(real_high)
        c_fake = self.C(fake_high)

        gradient_penalty = hp.gp_lambda * self._gp(real_high, fake_high, self.C)

        # Zero the gradients
        self.C_optimizer.zero_grad()


        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        # w_estimate = c_real_mean.detach().item() - c_fake_mean.detach().item()

        critic_loss.backward()

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
        fake_low = hp.low(hp.rf(fake))
        real_low = hp.low(hp.rf(fine))

        fake_high = fake - fake_low
        real_high = fine - real_low

        c_fake = self.C(fake_high)

        # Calculate generator loss
        g_loss = -torch.mean(c_fake)*hp.gamma

        # Add content loss and create objective function
        g_loss = g_loss + hp.content_lambda * content_loss(fake_low, real_low, device=config.device)

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
            retain_graph=True
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(hp.batch_size, -1).to(config.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return hp.gp_lambda * torch.mean((gradients_norm - 1) ** 2)


    def _train_epoch(self, dataloader, testdataloader, epoch):
        """
        Performs one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            epoch (int): The epoch number.
        """
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        print(f"Epoch {epoch}")
        print(80*"=")
        train_metrics = initialize_metric_dicts({})
        test_metrics = initialize_metric_dicts({})

        for i, data in enumerate(dataloader):
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
        post_epoch_metric_mean(train_metrics, "train")

        # Generate plots from training set
        cbatch, rbatch = next(iter(dataloader))
        gen_grid_images(self.G, cbatch, rbatch, epoch, "train")

        test_metrics = initialize_metric_dicts({})
        for i, data in enumerate(testdataloader):
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
