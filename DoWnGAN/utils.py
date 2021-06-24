from mlflow import log_param


def mlflow_dict_logger(d: dict):
    for key in d.keys():
        log_param(key, d[key])
