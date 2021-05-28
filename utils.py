from mlflow import log_metric, log_param, log_artifacts

def mlflow_dict_logger(d: dict):
	for key in d.keys():
		log_param(key, d[key]) 