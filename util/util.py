import hashlib
import json
import logging
import os
import time
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import *


def calc_index(predict, actual, threshold=0.5, log=True):
    """
    calculate f1 score by predict and actual.
    """
    if not isinstance(predict, torch.Tensor):
        predict = torch.as_tensor(predict)
    if not isinstance(actual, torch.Tensor):
        actual = torch.as_tensor(actual)

    predict = predict.detach().cpu()
    actual = actual.detach().cpu()

    if predict.dim() != 2:
        predict = predict.reshape(-1, predict.shape[-1])
    if actual.dim() != 2:
        actual = actual.reshape(-1, actual.shape[-1])

    actual_np = actual.numpy()
    predict_np = predict.numpy()
    try:
        ap = float(average_precision_score(actual_np, predict_np, average='macro'))
    except ValueError:
        ap = 0.0
    try:
        auc = float(roc_auc_score(actual_np, predict_np, average='macro'))
    except ValueError:
        auc = 0.0

    if predict.shape[-1] == 2 and actual.shape[-1] == 2:
        actual = torch.argmax(actual, dim=-1).numpy()
        if threshold is None:
            predict = torch.argmax(predict, dim=-1).numpy()
        else:
            anomaly_prob = predict[:, 1].numpy()
            predict = (anomaly_prob > float(threshold)).astype(np.int64)
    else:
        if actual.dim() == 2:
            actual = torch.argmax(actual, dim=-1).numpy()
        else:
            actual = actual.reshape(-1).numpy()
        if predict.dim() == 2:
            predict = torch.argmax(predict, dim=-1).numpy()
        else:
            predict = predict.reshape(-1).numpy()

    ps = float(precision_score(actual, predict, average="binary", zero_division=1))
    rs = float(recall_score(actual, predict, average="binary", zero_division=1))
    effection = float(f1_score(actual, predict, average="binary", zero_division=1))

    pred = np.bincount(predict.astype(np.int64), minlength=2)
    actu = np.bincount(actual.astype(np.int64), minlength=2)
    threshold_info = "argmax" if threshold is None else f"{float(threshold):.4f}"

    if pred.shape[0] == 1:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong: 0  actu_right: {actu[0]} actu_wrong: {actu[1]} thr:{threshold_info}'
    else:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong:{pred[1]} actu_right: {actu[0]} actu_wrong: {actu[1]} thr:{threshold_info}'
    if log:
        logging.info(information)
    return information, {'pr': ps, 'rc': rs, 'auc': auc, 'ap': ap, 'f1': effection, 'threshold': threshold}


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False, )


def dump_params(args):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")).hexdigest()[0:8]
    save_path = os.path.join(args['result_dir'], args['main_model'] + '-' +args['dataset_path'].split('/')[-1] + '-' + hash_id + '-'+ str(int(time.time())))
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id, save_path


def read_params(args):
    filename = os.path.join(args['model_path'], "params.json")
    with open(filename) as f:
        dict_json = json.load(fp=f)
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return dict_json


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)
