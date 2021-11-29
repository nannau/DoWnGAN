import stage as s
import hyperparams as hp
from DoWnGAN.wasserstein import WassersteinGAN
import mlflow
from mlflow_utils import log_hyperparams
import subprocess

# This runs mlflow server with the given backend
subprocess.run([
    "mlflow",
    "server", 
    "--host 0.0.0.0",
    "--backend-store-uri",
    hp.experiment_path, 
    "-p",
    "5555"
])

trainer = WassersteinGAN(
    s.generator,
    s.critic,
    s.G_optimizer,
    s.C_optimizer
)

mlflow.set_tracking_uri(hp.experiment_path)
print("Tracking URI: ", mlflow.get_tracking_uri())

with mlflow.start_run(experiment_id = s.exp_id, run_name = s.tag):
    mlflow.set_tag(s.tag.info.run_id, s.tag)
    log_hyperparams()
    trainer.train(
        s.dataloader, 
        s.testdataloader,
        epochs = hp.epochs,
    ) 