
from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.wasserstein import WassersteinGAN
from DoWnGAN.mlflow_tools.mlflow_utils import log_hyperparams

import mlflow
import torch
import subprocess

def train():
    s = StageData()
    torch.cuda.empty_cache()
    trainer = WassersteinGAN(
        s.generator,
        s.critic,
        s.G_optimizer,
        s.C_optimizer
    )

    mlflow.set_tracking_uri(config.EXPERIMENT_PATH)
    print("Tracking URI: ", mlflow.get_tracking_uri())

    with mlflow.start_run(experiment_id = s.exp_id, run_name = s.tag) as run:
        # mlflow.set_tag(run.info.run_id, s.tag)
        log_hyperparams()
        trainer.train(
            s.dataloader, 
            s.testdataloader,
        ) 

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # set up cluster and workers
    # dask.distributed.nanny["MALLOC_TRIM_THRESHOLD_"] = 0
    train()