from utils import mlflow_dict_logger
from losses import MedianPool2d, eof_loss, vorticity_loss, divergence_loss, content_loss, low_pass_eof_batch
import time
import logging
import subprocess
import tracemalloc

import numpy as np
from csv import DictWriter
import csv

from torch.nn.functional import avg_pool2d
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from gen_plots import generate_plots
from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import mlflow

import gc

from memory_profiler import profile

class Trainer():
    """Implementation of custom Wasserstein GAN with Gradient Penalty.
    See below for expected keys
    Args:
        models, dict: Dictionary containing critic and discriminator
        device, pytorch.cuda.device: GPU or CPU of host machine
        hyp_params, dict: Containing hyperparameters of run
        run_params, dict: Containing technical parameters and logging of run

    """
    def __init__(self, models: dict, hyper_params: dict, run_params: dict, experiment, tag):
        self.experiment = experiment
        self.tag = tag
        self.client = MlflowClient()

        self.G = models["generator"]
        self.C = models["critic"]
        self.G_opt = models["gen_optimizer"]
        self.C_opt = models["critic_optimizer"]
        self.low = low_pass_eof_batch

        self.hyper_params = hyper_params
        self.gp_weight = hyper_params["gp_weight"]
        self.critic_iterations = hyper_params["critic_iterations"]
        self.batch_size = hyper_params["batch_size"]
        self.gamma = hyper_params["gamma"]
        self.eof_weight = hyper_params["eof_weight"]
        self.div_weight = hyper_params["div_weight"]
        self.vort_weight = hyper_params["vort_weight"]
        self.content_weight = hyper_params["content_weight"]

        self.epochs = run_params["epochs"]
        self.print_every = run_params["print_every"]
        self.save_every = run_params["save_every"]
        self.use_cuda = run_params["use_cuda"]
        self.device = run_params["device"]

        self.num_steps = 0

        self.metrics = {}

        assert next(self.G.parameters()).is_cuda
        assert next(self.C.parameters()).is_cuda


    def _metric_log_every_n(self, name_of_metric, metric_to_log):
        if self.num_steps % self.save_every == 0 and self.num_steps > self.critic_iterations:
            log_metric(name_of_metric, metric_to_log)
            self._metric_print(name_of_metric, metric_to_log)

    def _metric_print(self, name_of_metric, metric_to_log):
        logging.basicConfig(format='%(asctime)s %(message)s')
        logging.info(f"{name_of_metric}: {metric_to_log}")

    def _critic_train_iteration(self, cr, hr, pca_og_shape, Z):
        """ """

        # Get generated data
        generated_data = self.G(cr)
        gen_low = self.low(Z.to(self.device), pca_og_shape.to(self.device), generated_data)
        high_pass = generated_data - gen_low
        high_pass_truth = hr - self.low(Z, pca_og_shape, hr)

        # Calculate probabilities on real and generated data
        # data = Variable(hr, requires_grad=True)
        data = Variable(high_pass_truth, requires_grad=True)

        d_real = self.C(data)
        # d_generated = self.C(generated_data)
        d_generated = self.C(high_pass)

        # Get gradient penalty
        # gradient_penalty = self._gradient_penalty(data, generated_data)
        gradient_penalty = self._gradient_penalty(data, high_pass)
        self.metrics["Gradient Penalty"] = gradient_penalty.item()

        # Create total loss and optimize
        self.C_opt.zero_grad()
        d_generated_mean = d_generated.mean()
        d_real_mean = d_real.mean()

        d_loss = d_generated_mean - d_real_mean + gradient_penalty
        w_estimate = -d_generated_mean + d_real_mean

        # Record loss
        self.metrics["Wasserstein Distance Estimate"] = w_estimate.item()

        d_loss.backward()
        self.C_opt.step()

        # Cleanup
        to_del = [d_loss, gradient_penalty, d_generated, d_real, data, d_generated_mean, d_real_mean, w_estimate, generated_data, high_pass_truth, high_pass, gen_low]
        for var in to_del:
            del var

    def _generator_train_iteration(self, cr, hr, pcas, pca_og_shape, Z):
        """ """
        self.G_opt.zero_grad()

        truth_low = self.low(Z.to(self.device), pca_og_shape.to(self.device), hr)


        # Get generated data
        generated_data = self.G(cr)

        gen_low = self.low(Z.to(self.device), pca_og_shape.to(self.device), generated_data)
        high_pass = generated_data - gen_low

        X = pcas[0, ...]
        eofloss = eof_loss(X, hr[:, :2, ...], generated_data[:, :2, ...], self.device)
        self.metrics["EOF Coefficient L2"] = eofloss
        
        divloss = divergence_loss(hr, generated_data, self.device)
        self.metrics["Divergence L2"] = divloss

        vortloss = vorticity_loss(hr, generated_data, self.device)
        self.metrics["Vorticity L2"] = vortloss

        # get content loss
        # cont_loss = content_loss(hr, generated_data, self.device)
        # self.metrics["Content L2"] = cont_loss.item()

        # Content loss on low pass
        cont_loss = content_loss(truth_low, gen_low, self.device)
        self.metrics["Content L2"] = cont_loss.item()

        # Calculate loss and optimize
        # d_generated = self.C(generated_data)
        # High freq only
        d_generated = self.C(high_pass)
        g_loss = - d_generated.mean()

        # g_loss = (
        #     self.eof_weight*eofloss + 
        #     self.vort_weight*vortloss + 
        #     self.div_weight*divloss + 
        #     self.content_weight*cont_loss + 
        #     self.gamma*g_loss
        # )

        g_loss = self.eof_weight*eofloss + self.content_weight*cont_loss + self.gamma*g_loss

        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.metrics["Generator loss"] = g_loss.item()

        to_del = [g_loss, cont_loss, eofloss, vortloss, divloss, d_generated, generated_data, X, gen_low, truth_low, high_pass]
        for var in to_del:
            del var

    def _gradient_penalty(self, real_data, generated_data):
        # batch_size = self.batch_size
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.C(interpolated)#.to(self.device)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1).to(self.device)
        # gnorm = gradients.norm(2, dim=1).mean().detach().cpu()
        # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().detach().cpu())
        # metric_log_every_n("Gradient Norm L2", gnorm)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # self._metric_log_every_n("Gradient Norm L2", gradients_norm.mean().item())#detach().cpu())

        to_del = [gradients, prob_interpolated, interpolated, alpha]
        for var in to_del:
            del var

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()


    def _train_epoch(self, data_loader, fixed, epoch):
        for i, data in enumerate(data_loader):

            hr = data[0].clone().detach().requires_grad_(True).to(self.device)
            # hr = torch.cat((hr, torch.randn(hr.size(0), 1, hr.size(2), hr.size(3)).to(self.device)), dim=1) # Add for stochasticity
            # low_hr = data[1].clone().detach().requires_grad_(True).to(self.device)
            # high_hr = data[2].clone().detach().requires_grad_(True).to(self.device)

            cr = data[1].clone().detach().requires_grad_(True).to(self.device)
            # cr = torch.cat((cr, torch.randn(cr.size(0), 1, cr.size(2), cr.size(3)).to(self.device)), dim=1) # Add for stochasticity
            pcas = data[2].clone().detach().requires_grad_(True).to(self.device)

            pca_og_shape = data[3].clone().detach().requires_grad_(True).to(self.device)

            Z = data[4].clone().detach().requires_grad_(True).to(self.device)

            self.num_steps += 1
            self._critic_train_iteration(cr, hr, pca_og_shape, Z)

            # result = subprocess.check_output(['bash','-c', 'free -m'])
            # free_memory = float(result.split()[9])
            # self.metrics["System Free Memory"] = free_memory
            self.metrics["Iteration number"] = self.num_steps
            self.metrics["GPU Memory"] = torch.cuda.memory_allocated()

            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(cr, hr, pcas, pca_og_shape, Z)

            if self.num_steps == 7:
                with open("metrics.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.metrics.keys())
                    f.close()

            if self.num_steps % self.save_every == 0:
                fixed_writer = {}
                fixed_writer["fake"] = self.G(fixed["coarse"]).detach().cpu()
                fixed_writer["real"] = fixed["fine"].detach().cpu()
                fixed_writer["coarse"] = fixed["coarse"].detach().cpu()
                # fixed_writer["fake"] = self.G(cr).detach().cpu()
                # fixed_writer["real"] = hr.detach().cpu()
                # fixed_writer["coarse"] = cr.detach().cpu()
                generate_plots(fixed_writer)

                for key in self.metrics.keys():
                    self._metric_log_every_n(key, self.metrics[key])

                with open("metrics.csv", "a") as f:
                    metric_file = DictWriter(f, fieldnames=self.metrics.keys())
                    metric_file.writerow(self.metrics)
                    f.close()

                mlflow.log_artifact("metrics.csv")

                # Log model
                mlflow.pytorch.log_model(self.C, "Critic")
                mlflow.pytorch.log_state_dict(self.C.state_dict(), "Critic")                
                mlflow.pytorch.log_model(self.G, "Generator")
                mlflow.pytorch.log_state_dict(self.G.state_dict(), "Generator")
                del fixed_writer

            to_del = [hr, cr, pcas]
            for var in to_del:
                del var

    def train(self, data_loader, epochs, fixed):
        logging.basicConfig(level=logging.INFO)

        # Fix sample to see how image generation improves during training
        fixed_coarse = Variable(fixed["coarse"], requires_grad=False)
        fixed_real = Variable(fixed["fine"], requires_grad=False)

        if self.use_cuda:
            fixed_coarse = fixed_coarse.to(self.device)
            fixed_real = fixed_real.to(self.device)

        fixed_writer = {"coarse": fixed_coarse, "fine": fixed_real}

        experiment_id = self.client.get_experiment_by_name(self.experiment).experiment_id
        

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tag(run.info.run_id, self.tag)
            mlflow_dict_logger(self.hyper_params)


            for epoch in range(self.epochs):
                logging.basicConfig(format='%(asctime)s %(message)s')
                logging.info(f'\nEpoch {epoch}')

                tracemalloc.start()
                self._train_epoch(data_loader, fixed_writer, epoch)
                current, peak = tracemalloc.get_traced_memory()
                self.metrics["System Free Memory Peak MB"] = peak/10**6
                logging.info(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

                tracemalloc.stop()
                gc.collect()
