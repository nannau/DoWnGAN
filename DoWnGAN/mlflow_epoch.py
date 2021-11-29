# Calculates epoch losses and logs them
import csv
import mlflow
from DoWnGAN.losses import content_loss, content_MSELoss, SSIM_Loss
import hyperparams as hp
import stage as s
import torch
import os
import pandas as pd

from csv import DictWriter

mlflow.set_tracking_uri(hp.experiment_path)

def log_to_file(dict, train_test):
    """Writes the metrics to a csv file"""
    csv_path = f"{mlflow.get_artifact_uri}/{train_test}_metrics.csv"
    # This will write to a new csv file if there isn't one
    # but append to an existing one if there is one
    with open(csv_path, "a", newline="") as f:
        df = pd.DataFrame.from_dict(data=dict)
        df.to_csv(f, header=(f.tell()==0))
    mlflow.log_artifact(csv_path)


def initialize_metric_dicts(d):
    for key in hp.metrics_to_calculate.keys():
        d[key] = []


def post_epoch_metric_mean(d, train_test):
    # Tracks batch metrics through 
    means = {}
    for key in hp.metrics_to_calculate.keys():
        means[key] = torch.mean(
            torch.FloatTensor(d[key])
        )
    
    log_to_file(means, train_test)


def gen_batch_and_log_metrics(G, C, coarse, real, d):
    fake = G(coarse)
    creal = torch.mean(C(real))
    cfake = torch.mean(C(fake))

    for key in hp.metrics_to_calculate.keys():
        if key == "Wass":
            d[key].append(hp.metrics_to_calculate[key](creal, cfake))
        else:
            d[key].append(hp.metrics_to_calculate[key](real, fake))