from DoWnGAN.utils import mlflow_dict_logger, metric_print
from DoWnGAN.losses import content_loss, low_pass_eof_batch, content_MSELoss, SSIM_Loss
import logging

from csv import DictWriter
import csv

from torch.autograd import grad as torch_grad

import torch
import torch.nn as nn

from DoWnGAN.gen_plots import generate_plots
from mlflow import log_metric

from mlflow.tracking import MlflowClient
import mlflow
from pkg_resources import resource_filename
from scipy.ndimage import gaussian_filter
# from skimage.metrics import structural_similarity as ssim
import torchgeometry as tgm
from torchgeometry.losses import ssim

class Trainer:
    """Implementation of custom Wasserstein GAN with Gradient Penalty.
    See below for expected keys
    Args:
        models, dict: Dictionary containing critic and discriminator
        device, pytorch.cuda.device: GPU or CPU of host machine
        hyp_params, dict: Containing hyperparameters of run
        run_params, dict: Containing technical parameters and logging of run

    """

    def __init__(
        self,
        models: dict,
        hyper_params: dict,
        run_params: dict,
        experiment,
        tag,
    ):
        self.experiment_id = experiment
        self.tag = tag
        self.client = MlflowClient()

        self.G = models["generator"]
        self.C = models["critic"]
        self.G_opt = models["gen_optimizer"]
        self.C_opt = models["critic_optimizer"]

        self.padding = 2
        self.low = nn.AvgPool2d(5, stride=1, padding=0)
        # self.low = tgm.image.GaussianBlur((15, 15), (3, 3))j
        self.rf = nn.ReplicationPad2d(self.padding)
        self.hyper_params = hyper_params
        self.hyper_params["padding"] = self.padding
        self.hyper_params["average_pool_layer"] = str(self.low._get_name)
        self.run_params = run_params

        self.device = run_params["device"]
        mlflow.set_tracking_uri(self.run_params["log_path"])

        self.num_steps = 0
        self.metrics = {}

        assert next(self.G.parameters()).is_cuda
        assert next(self.C.parameters()).is_cuda

    def _critic_train_iteration(self, cr, hr, cr_v, hr_v):
        """ """
        # Get generated data
        generated_data = self.G(cr)

        gen_low = self.low(self.rf(generated_data))

        true_low = self.low(self.rf(hr))
        # true_low = self.low(Z, EOFs, hr, transformer, self.device)

        # Calculate high frequencies
        high_pass = generated_data - gen_low
        high_pass_truth = hr - true_low


        # Calculate probabilities on real and generated data
        d_real = self.C(high_pass_truth)
        # d_real_v = self.C(hr_v)


        d_generated = self.C(high_pass)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(high_pass_truth, high_pass, self.C)
        self.metrics["Gradient Penalty"] = gradient_penalty.item()

        # Create total loss and optimize
        self.C_opt.zero_grad()

        d_generated_mean = torch.mean(d_generated)

        d_real_mean = torch.mean(d_real)


        d_loss = (d_generated_mean - d_real_mean) + gradient_penalty
        w_estimate = -d_generated_mean + d_real_mean

        d_loss.backward(retain_graph=True)

        # Record loss
        self.metrics["Wasserstein Distance Estimate"] = w_estimate.item()

        if self.num_steps % self.run_params["save_every"] == 0:

            generated_data_v = self.G(cr_v)
            gen_low_v = self.low(self.rf(generated_data_v))
            high_pass_v = generated_data_v - gen_low_v
            high_pass_truth_v = hr_v - self.low(self.rf(hr_v))
            d_real_v = self.C(high_pass_truth_v)
            d_generated_v = self.C(high_pass_v)
            d_generated_mean_v = torch.mean(d_generated_v)
            d_real_mean_v = torch.mean(d_real_v)
            w_estimate_v = -d_generated_mean_v + d_real_mean_v

            self.metrics["Validation Wasserstein Distance Estimate"] = w_estimate_v.item()


        self.C_opt.step()


    def _generator_train_iteration(self, cr, hr, cr_v, hr_v):
        """ """
        self.G_opt.zero_grad()
        truth_low = self.low(self.rf(hr))

        # Get generated data
        generated_data = self.G(cr)

        gen_low = self.low(self.rf(generated_data))

        high_pass = generated_data - gen_low

        # Content loss on low pass 
        cont_loss = content_loss(truth_low, gen_low, self.device)
        cont_loss_all_freq = content_loss(hr, generated_data, self.device)

        mse_loss = content_MSELoss(truth_low, gen_low, self.device)
        mse_loss_all_freq = content_MSELoss(hr, generated_data, self.device)

        # self.metrics["Content L1"] = cont_loss.item()
        # cont_loss = content_loss(generated_data, hr, self.device)
        # cont_loss_v = content_loss(generated_data_v, hr_v, self.device)

        self.metrics["Content MSE Low Freq"] = mse_loss.item()
        self.metrics["Content MSE All Freq"] = mse_loss_all_freq.item()

        self.metrics["Content L1 Low Freq"] = cont_loss.item()
        self.metrics["Content L1 All Freq"] = cont_loss_all_freq.item()

        ss_all = SSIM_Loss(generated_data.clone(), hr.clone())
        self.metrics["MS-SSIM All Freq"] = 1. - ss_all.item()
        ss_low = SSIM_Loss(truth_low.clone(), gen_low.clone())
        self.metrics["MS-SSIM Low Freq"] = 1. - ss_low.item()


        if self.num_steps % 50 == 0:
            truth_low_v = self.low(self.rf(hr_v))
            generated_data_v = self.G(cr_v)
            gen_low_v = self.low(self.rf(generated_data_v))
            cont_loss_all_freq_v = content_loss(hr_v, generated_data_v, self.device)
            cont_loss_v = content_loss(gen_low_v, truth_low_v, self.device)

            ss_all_v = SSIM_Loss(generated_data_v.clone(), hr_v.clone())
            self.metrics["Validation MS-SSIM All Freq"] = 1. - ss_all_v.item()
            ss_low_v = SSIM_Loss(truth_low_v.clone(), gen_low_v.clone())
            self.metrics["Validation MS-SSIM Low Freq"] = 1. - ss_low_v.item()


            self.metrics["Validation Content L1 Low Freq"] = cont_loss_v.item()
            self.metrics["Validation Content L1 All Freq"] = cont_loss_all_freq_v.item()

            mse_loss_v = content_MSELoss(truth_low_v, gen_low_v, self.device)
            mse_loss_all_freq_v = content_MSELoss(hr_v, generated_data_v, self.device)

            # self.metrics["Content L1"] = cont_loss.item()
            # cont_loss = content_loss(generated_data, hr, self.device)
            # cont_loss_v = content_loss(generated_data_v, hr_v, self.device)

            self.metrics["Validation Content MSE Low Freq"] = mse_loss_v.item()
            self.metrics["Validation Content MSE All Freq"] = mse_loss_all_freq_v.item()



        # Calculate loss and optimize
        # High freq only
        d_generated = self.C(high_pass)

        g_loss = -torch.mean(d_generated)
        g_loss = (
                # ss_all
                + self.hyper_params["content_weight"] * cont_loss
                # + self.hyper_params["content_weight"] * (1. - ss_all.item())
                + self.hyper_params["gamma"] * g_loss
        )
        g_loss.backward()

        self.G_opt.step()

        # Record loss
        self.metrics["Generator loss"] = g_loss.item()


    def _gradient_penalty(self, real_data, generated_data, critic):

        current_batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        prob_interpolated = critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(self.hyper_params["batch_size"], -1).to(self.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.hyper_params["gp_weight"] * torch.mean((gradients_norm - 1) ** 2)

    def _gradient_penalty_low(self, real_data, generated_data, critic):

        current_batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        prob_interpolated = critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(self.hyper_params["batch_size"], -1).to(self.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.hyper_params["gp_weight"] * torch.mean((gradients_norm - 1) ** 2)

    def _train_epoch(self, data_loader, fixed, epoch, validate):
        # hr = iter(data_loader).next()[0].clone().detach().requires_grad_(True).to(self.device).float()
        # cr = iter(data_loader).next()[0].clone().detach().requires_grad_(True).to(self.device).float()

        hr_v = validate["fine"].to(self.device).float()
        cr_v = validate["coarse"].to(self.device).float()
        for i, data in enumerate(data_loader):
            hr = data[0].to(self.device)#clone().detach().requires_grad_(True).to(self.device).float()
            cr = data[1].to(self.device)#clone().detach().requires_grad_(True).to(self.device).float()

            self.num_steps += 1
            self._critic_train_iteration(cr, hr, cr_v, hr_v)
            self.metrics["Iteration number"] = self.num_steps

            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.hyper_params["critic_iterations"] == 0:
                self._generator_train_iteration(cr, hr, cr_v, hr_v)

            # if self.num_steps == 1 + self.hyper_params["critic_iterations"]:
        # if self.num_steps == self.run_params["save_every"]:
        if epoch == 0:
            with open("metrics.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.metrics.keys())
                f.close()

        # if self.num_steps % self.run_params["save_every"] == 0:
        fixed_writer = {}
        fixed_writer["fake"] = self.G(fixed["coarse"].to(self.device)).detach().cpu()
        fixed_writer["real"] = fixed["fine"].detach().cpu()
        fixed_writer["coarse"] = fixed["coarse"].detach().cpu()

        fixed_writer["low_real"] = self.low(fixed["fine"][:3, ...]).detach().cpu()
        fixed_writer["low_fake"] = self.low(self.G(fixed["coarse"][:3, ...].to(self.device))).detach().cpu()

        generate_plots(fixed_writer, self.artifacts_uri)

        [log_metric(key, self.metrics[key]) for key in self.metrics.keys()]
        [metric_print(key, self.metrics[key]) for key in self.metrics.keys()]

        with open("metrics.csv", "a") as f:
            metric_file = DictWriter(f, fieldnames=self.metrics.keys())
            metric_file.writerow(self.metrics)
            f.close()

        # Log model
        print(f"ARTIFACT URI: {mlflow.get_artifact_uri()}")
        mlflow.log_artifact("metrics.csv")#, self.artifacts_uri+"/metrics.csv")
        mlflow.pytorch.log_model(self.C, f"Critic/Critic_{epoch}")
        mlflow.pytorch.log_state_dict(self.C.state_dict(), f"Critic/Critic_{epoch}")
        mlflow.pytorch.log_model(self.G, f"Generator/Generator_{epoch}")
        mlflow.pytorch.log_state_dict(self.G.state_dict(), f"Generator/Generator_{epoch}")


    def train(self, data_loader, epochs, fixed, validate):
        self.num_steps = 0
        logging.basicConfig(level=logging.INFO)
        mlflow.set_tracking_uri(self.run_params["log_path"])
        with mlflow.start_run(experiment_id=self.experiment_id, run_name = self.tag) as run:
            self.run = run
            self.artifacts_uri = mlflow.get_artifact_uri()
            mlflow.set_tag(run.info.run_id, self.tag)
            mlflow_dict_logger(self.hyper_params)

            for epoch in range(self.run_params["epochs"]):
                logging.basicConfig(format="%(asctime)s %(message)s")
                logging.info(f"\nEpoch {epoch}")

                self._train_epoch(data_loader, fixed, epoch, validate)
