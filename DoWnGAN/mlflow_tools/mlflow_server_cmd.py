from DoWnGAN.config import config
import subprocess

subprocess.run(["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", config.EXPERIMENT_PATH, "-p", "5555"])