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


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def prepare_binary_classification_inputs(predict, actual):
    predict = _to_numpy(predict)
    actual = _to_numpy(actual)

    if predict.ndim > 2:
        predict = predict.reshape(-1, predict.shape[-1])
    if actual.ndim > 2:
        actual = actual.reshape(-1, actual.shape[-1])

    if predict.ndim == 2 and predict.shape[-1] >= 2:
        score = predict[:, 1].astype(float)
    else:
        score = predict.reshape(-1).astype(float)

    if actual.ndim == 2 and actual.shape[-1] >= 2:
        label = np.argmax(actual, axis=-1).astype(int)
    else:
        label = actual.reshape(-1).astype(int)

    return score.reshape(-1), label.reshape(-1)


def find_best_f1_threshold(score, label, default_threshold=0.5):
    score = np.asarray(score, dtype=float).reshape(-1)
    label = np.asarray(label, dtype=int).reshape(-1)

    if score.size == 0 or label.size == 0:
        return float(default_threshold)
    if len(np.unique(label)) < 2:
        return float(default_threshold)

    precision, recall, thresholds = precision_recall_curve(label, score)
    if thresholds.size == 0:
        return float(default_threshold)

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def calc_binary_score_metrics(score, label, threshold=0.5, zero_division=1, raw_predict=None, raw_actual=None):
    score = np.asarray(score, dtype=float).reshape(-1)
    label = np.asarray(label, dtype=int).reshape(-1)
    threshold = float(threshold)
    pred = (score >= threshold).astype(int)

    if raw_predict is not None and raw_actual is not None:
        raw_predict = _to_numpy(raw_predict)
        raw_actual = _to_numpy(raw_actual)
        if raw_predict.ndim > 2:
            raw_predict = raw_predict.reshape(-1, raw_predict.shape[-1])
        if raw_actual.ndim > 2:
            raw_actual = raw_actual.reshape(-1, raw_actual.shape[-1])
        try:
            ap = float(average_precision_score(raw_actual, raw_predict, average='macro'))
        except ValueError:
            ap = float("nan")
        try:
            auc = float(roc_auc_score(raw_actual, raw_predict, average='macro'))
        except ValueError:
            auc = float("nan")
    else:
        try:
            ap = float(average_precision_score(label, score))
        except ValueError:
            ap = float("nan")
        try:
            auc = float(roc_auc_score(label, score))
        except ValueError:
            auc = float("nan")

    ps = float(precision_score(label, pred, average="binary", zero_division=zero_division))
    rs = float(recall_score(label, pred, average="binary", zero_division=zero_division))
    effection = float(f1_score(label, pred, average="binary", zero_division=zero_division))

    pred_count = np.bincount(pred, minlength=2)
    actu_count = np.bincount(label, minlength=2)

    return {
        'pr': ps,
        'rc': rs,
        'auc': auc,
        'ap': ap,
        'f1': effection,
        'threshold': threshold,
        'pred_right': int(pred_count[0]),
        'pred_wrong': int(pred_count[1]),
        'actu_right': int(actu_count[0]),
        'actu_wrong': int(actu_count[1]),
    }


def format_binary_metrics(metrics):
    return (
        f"pr:{metrics['pr']:.4f}  rc:{metrics['rc']:.4f}  auc:{metrics['auc']:.4f} "
        f"ap:{metrics['ap']:.4f} f1: {metrics['f1']:.4f} "
        f"pred_right: {metrics['pred_right']} pred_wrong:{metrics['pred_wrong']} "
        f"actu_right: {metrics['actu_right']} actu_wrong: {metrics['actu_wrong']}"
    )


def calc_index(predict, actual):
    """
    calculate f1 score by predict and actual.
    """
    score, label = prepare_binary_classification_inputs(predict, actual)
    metrics = calc_binary_score_metrics(score, label, threshold=0.5)
    information = format_binary_metrics(metrics)
    logging.info(information)
    return information, {key: metrics[key] for key in ['pr', 'rc', 'auc', 'ap', 'f1']}


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
