# Calculates epoch losses and logs them
from DoWnGAN.GAN.losses import content_loss, content_MSELoss, SSIM_Loss
from DoWnGAN.config import config
import DoWnGAN.config.hyperparams as hp
import DoWnGAN.GAN.stage as s

import csv
import mlflow
from mlflow import log_param, log_metric

import torch
import os
import pandas as pd

from csv import DictWriter

mlflow.set_tracking_uri(config.EXPERIMENT_PATH)

def log_to_file(dict, train_test):
    """Writes the metrics to a csv file"""
    csv_path = f"{mlflow.get_artifact_uri()}/{train_test}_metrics.csv"
    # This will write to a new csv file if there isn't one
    # but append to an existing one if there is one
    with open(csv_path, "a", newline="") as f:
        df = pd.DataFrame.from_dict(data=dict)
        df.to_csv(f, header=(f.tell()==0))
    # mlflow.log_artifact(csv_path)


def initialize_metric_dicts(d):
    for key in hp.metrics_to_calculate.keys():
        d[key] = []
    return d


def metric_print(metric, metric_value):
    print(f"{metric}: {metric_value}")


def post_epoch_metric_mean(d, train_test):
    # Tracks batch metrics through 
    means = {}
    for key in hp.metrics_to_calculate.keys():
        means[key] = [torch.mean(
            torch.FloatTensor(d[key])
        ).item()]
        log_metric(f"{key}_{train_test}", means[key][0])
        metric_print(f"{key}_{train_test}", means[key][0])

    log_to_file(means, train_test)


def gen_batch_and_log_metrics(G, C, coarse, real, d):
    fake = G(coarse).detach()
    creal = torch.mean(C(real)).detach()
    cfake = torch.mean(C(fake)).detach()

    for key in hp.metrics_to_calculate.keys():
        if key == "Wass":
            d[key].append(hp.metrics_to_calculate[key](creal, cfake, config.device).detach().cpu().item())
        else:
            d[key].append(hp.metrics_to_calculate[key](real, fake, config.device).detach().cpu().item())
    return d

def log_network_models(C, G, epoch):
    mlflow.pytorch.log_model(C, f"Critic/Critic_{epoch}")
    mlflow.pytorch.log_state_dict(C.state_dict(), f"Critic/Critic_{epoch}")
    mlflow.pytorch.log_model(G, f"Generator/Generator_{epoch}")
    mlflow.pytorch.log_state_dict(G.state_dict(), f"Generator/Generator_{epoch}")