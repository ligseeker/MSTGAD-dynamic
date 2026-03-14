import copy
import csv
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief
from tqdm import tqdm

import util.util as util


class Base(nn.Module):
    def __init__(self, model, **args):
        super(Base, self).__init__()

        self.model = model
        self.use_gpu = args['gpu']
        self.device = torch.device('cuda' if args['gpu'] and torch.cuda.is_available() else 'cpu')

        self.epoches = args['epochs']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']
        self.patience = args['patience']
        self.model_save_dir = args['result_dir']
        self.learning_change = args['learning_change']
        self.learning_gamma = args['learning_gamma']
        self.eval_interval = max(1, int(args.get('eval_interval', 1)))
        self.train_eval_interval = max(0, int(args.get('train_eval_interval', 1)))
        self.rec_down = args['rec_down']
        self.para_low = args['para_low']
        self.True_list = {'normal': 1, 'abnormal': args['abnormal_weight']}

        self.split_mode = args.get('split_mode', 'baseline70_30')
        self.score_threshold_mode = args.get('score_threshold_mode', 'val_f1')
        self.manual_score_threshold = args.get('score_threshold')
        self.best_metric_name = args.get('best_metric', 'val_f1')
        self.early_stop_metric_name = args.get('early_stop_metric', 'val_f1')
        self.early_stop_patience = max(0, int(args.get('early_stop_patience', 6)))

        self.metrics_history_path = os.path.join(self.model_save_dir, 'metrics_history.csv')
        self.best_metrics_path = os.path.join(self.model_save_dir, 'best_metrics.json')
        self.best_records = {
            'loss': {'score': float("inf"), 'state': None, 'epoch': -1},
            'f1': {'score': float("-inf"), 'state': None, 'epoch': -1},
        }

        if not args['evaluate']:
            self._init_metrics_history()

        if args['evaluate']:
            self.load_model(args['model_path'])
        else:
            logging.info('model : init weight')
            self.init_weight()

        if self.device.type == 'cuda':
            logging.info("Using GPU...")
            torch.cuda.empty_cache()
            self.model.cuda()
        else:
            logging.info("Using CPU...")

    def _init_metrics_history(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        with open(self.metrics_history_path, 'w', newline='') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    'epoch',
                    'split',
                    'pr',
                    'rc',
                    'auc',
                    'ap',
                    'f1',
                    'threshold',
                    'checkpoint_type',
                ],
            )
            writer.writeheader()

    def _append_metrics_history(self, epoch, split, metrics, checkpoint_type='latest'):
        if not os.path.exists(self.metrics_history_path):
            return

        with open(self.metrics_history_path, 'a', newline='') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    'epoch',
                    'split',
                    'pr',
                    'rc',
                    'auc',
                    'ap',
                    'f1',
                    'threshold',
                    'checkpoint_type',
                ],
            )
            writer.writerow({
                'epoch': epoch,
                'split': split,
                'pr': f"{metrics['pr']:.6f}",
                'rc': f"{metrics['rc']:.6f}",
                'auc': f"{metrics['auc']:.6f}",
                'ap': f"{metrics['ap']:.6f}",
                'f1': f"{metrics['f1']:.6f}",
                'threshold': f"{metrics['threshold']:.6f}",
                'checkpoint_type': checkpoint_type,
            })

    def init_weight(self):
        for p in self.model.parameters():
            if p.dim() > 1 and all(s > 0 for s in p.shape):
                nn.init.xavier_uniform_(p)

    def input2device(self, batch_input, use_gpu):
        def _to_float_tensor(data):
            tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            return tensor.to(self.device, dtype=torch.float32, non_blocking=(self.device.type == 'cuda'))

        if isinstance(batch_input, dict):
            for name, data in batch_input.items():
                batch_input[name] = _to_float_tensor(data)
        else:
            batch_input = _to_float_tensor(batch_input)
        return batch_input

    def load_model(self, model_save_file="", name='loss'):
        if model_save_file == ' ':
            logging.info(f'No {self.model.name} statue file')
            return False

        logging.info(f'{self.model.name} on {model_save_file} loading...')
        ckpt_path = os.path.join(model_save_file, f"{self.model.name}_{name}_stage.ckpt")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            return True

        logging.info(f"{ckpt_path} not found, skipping load.")
        return False

    def save_model(self, best_dict, model_save_dir="", name='loss'):
        os.makedirs(model_save_dir, exist_ok=True)
        file_status = os.path.join(model_save_dir, f"{self.model.name}_{name}_stage.ckpt")
        if best_dict['state'] is None:
            logging.info(f'No {self.model.name} - {name} statue file')
        else:
            logging.info(f'{self.model.name} - {name}  best score:{best_dict["score"]} at epoch {best_dict["epoch"]}')
            torch.save(best_dict['state'], file_status)


class MY(Base):
    def __init__(self, model, **args):
        super().__init__(model, **args)
        self.contrast_weight = args.get('contrast_weight', 0.1)
        self.contrast_warmup = args.get('contrast_warmup', 5)
        self.contrast_start_epoch = max(0, int(args.get('contrast_start_epoch', self.contrast_warmup)))
        self.graph_update_steps = max(1, int(args.get('graph_update_steps', 4)))
        self.graph_summary_mode = args.get('graph_summary_mode', 'last')
        self.contrast_summary_mode = args.get('contrast_summary_mode', 'last')
        self.score_fusion_alpha = float(args.get('score_fusion_alpha', 1.0))

    def _contrast_gamma(self, epoch):
        if self.contrast_weight <= 0 or epoch < self.contrast_start_epoch:
            return 0.0
        if self.contrast_warmup <= 0:
            return self.contrast_weight
        progress = min((epoch - self.contrast_start_epoch + 1) / self.contrast_warmup, 1.0)
        return self.contrast_weight * progress

    def _default_threshold(self):
        if self.score_threshold_mode == 'manual' and self.manual_score_threshold is not None:
            return float(self.manual_score_threshold)
        return 0.5

    def _metric_key(self, metric_name):
        metric_name = (metric_name or 'f1').lower()
        if '_' in metric_name:
            return metric_name.split('_')[-1]
        return metric_name

    def _metric_value(self, metrics, metric_name):
        key = self._metric_key(metric_name)
        return float(metrics.get(key, float('-inf')))

    def _should_eval(self, epoch, interval):
        if interval <= 0:
            return False
        if epoch == self.epoches - 1:
            return True
        if epoch < self.rec_down:
            return False
        return (epoch + 1) % interval == 0

    def _collect_predictions(self, data_loader):
        self.model.eval()
        if hasattr(self.model, 'reset_dynamic_graph_cache'):
            self.model.reset_dynamic_graph_cache(reset_stats=True)

        with torch.no_grad():
            predict_list, label_list, rec_score_list = [], [], []
            for batch_input in tqdm(data_loader):
                batch_input = self.input2device(batch_input, self.use_gpu)
                if self.score_fusion_alpha < 1.0:
                    raw_result, _, rec_score = self.model(
                        batch_input, evaluate=True, return_eval_aux=True
                    )
                    rec_score_list.append(rec_score)
                else:
                    raw_result, _ = self.model(batch_input, evaluate=True)

                predict_list.append(raw_result)
                label_list.append(batch_input['groundtruth_real'])

        predict_list = torch.concat(predict_list, dim=0).cpu()
        label_list = torch.concat(label_list, dim=0).cpu()

        if self.score_fusion_alpha < 1.0 and rec_score_list:
            rec_score_list = torch.concat(rec_score_list, dim=0).reshape(-1).cpu()
            predict_list = self._fuse_predict_with_reconstruction(predict_list, rec_score_list)

        score, label = util.prepare_binary_classification_inputs(predict_list, label_list)
        return score, label, predict_list, label_list

    def evaluate(
        self,
        data_loader,
        split='test',
        threshold=None,
        calibrate_threshold=False,
        log_metrics=True,
        epoch=None,
        checkpoint_type='latest',
    ):
        score, label, raw_predict, raw_actual = self._collect_predictions(data_loader)

        if calibrate_threshold and self.score_threshold_mode == 'val_f1':
            threshold = util.find_best_f1_threshold(score, label, default_threshold=self._default_threshold())
        elif threshold is None:
            threshold = self._default_threshold()

        metrics = util.calc_binary_score_metrics(
            score,
            label,
            threshold=threshold,
            raw_predict=raw_predict,
            raw_actual=raw_actual,
        )
        info = util.format_binary_metrics(metrics)

        if log_metrics:
            logging.info(f"[{split}] {info}")
        if epoch is not None:
            self._append_metrics_history(epoch, split, metrics, checkpoint_type=checkpoint_type)

        return {'info': info, 'metrics': metrics}

    def fit(self, train_loader, eval_loader, eval_split='test', test_loader=None, **args):
        optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.learning_change, self.learning_gamma)

        label_weight = torch.tensor(
            np.array(list(self.True_list.values())), dtype=torch.float, device=self.device
        )
        losser = nn.BCEWithLogitsLoss(reduction='mean', weight=label_weight)

        logging.info('optimizer : using AdaBelief')
        logging.info('Innovations: Dynamic Graph + Gated Fusion + Contrastive Learning')
        logging.info(
            f'  contrast_weight={self.contrast_weight}, contrast_start_epoch={self.contrast_start_epoch}, '
            f'contrast_warmup={self.contrast_warmup}'
        )
        logging.info(
            f'  graph_update_steps={self.graph_update_steps}, graph_summary_mode={self.graph_summary_mode}'
        )
        logging.info(
            f'  contrast_summary_mode={self.contrast_summary_mode}, score_fusion_alpha={self.score_fusion_alpha}'
        )
        logging.info(
            f'  split_mode={self.split_mode}, eval_interval={self.eval_interval}, '
            f'train_eval_interval={self.train_eval_interval}, precision=float32'
        )

        global_step = 0
        bad_eval_rounds = 0
        best_early_stop_score = float("-inf")

        for epoch in range(0, self.epoches):
            lr = optimizer.param_groups[0]['lr']
            para = torch.tensor(1 / (epoch // self.rec_down + 1), device=self.device)
            para = para if para > self.para_low else self.para_low
            gamma = self._contrast_gamma(epoch)
            compute_contrast = gamma > 0

            if hasattr(self.model, 'reset_dynamic_graph_cache'):
                self.model.reset_dynamic_graph_cache(reset_stats=True)

            logging.info('-' * 100)
            logging.info(
                f'{epoch}/{self.epoches} starting... lr: {lr} para:{para} gamma(contrast):{gamma:.4f}'
            )

            self.model.train()
            epoch_cls_loss, epoch_rec_loss, epoch_loss = [], [], []
            epoch_contrast_loss, epoch_graph_reg_loss = [], []
            epoch_time_start = time.time()
            is_wrong = False

            with tqdm(train_loader) as tbar:
                for batch_input in tbar:
                    batch_input = self.input2device(batch_input, self.use_gpu)
                    optimizer.zero_grad()

                    raw_loss, cls_result, cls_label, contrast_loss, graph_reg_loss = self.model(
                        batch_input,
                        global_step=global_step,
                        compute_contrast=compute_contrast,
                    )

                    rec_loss = sum(raw_loss)
                    if cls_result.shape[0] == 0:
                        cls_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    else:
                        cls_loss = losser(cls_result, cls_label)

                    loss = (1 - para) * cls_loss + para * rec_loss + gamma * contrast_loss + graph_reg_loss

                    if not torch.isfinite(loss):
                        is_wrong = True
                        logging.info("loss is not finite")
                        break

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                    optimizer.step()
                    global_step += 1

                    epoch_cls_loss.append(cls_loss.item())
                    epoch_rec_loss.append(rec_loss.item())
                    epoch_contrast_loss.append(contrast_loss.item())
                    epoch_graph_reg_loss.append(graph_reg_loss.item())
                    epoch_loss.append(loss.item())
                    tbar.set_postfix(
                        loss=f'{loss.item():.6f}',
                        cls=f'{cls_loss.item():.6f}',
                        rec=f'{rec_loss.item():.6f}',
                        ctr=f'{contrast_loss.item():.4f}',
                    )

            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = torch.mean(torch.tensor(epoch_loss)).item() if epoch_loss else float("inf")
            epoch_cls_loss = torch.mean(torch.tensor(epoch_cls_loss)).item() if epoch_cls_loss else 0.0
            epoch_rec_loss = torch.mean(torch.tensor(epoch_rec_loss)).item() if epoch_rec_loss else 0.0
            epoch_contrast_loss = torch.mean(torch.tensor(epoch_contrast_loss)).item() if epoch_contrast_loss else 0.0
            epoch_graph_reg_loss = torch.mean(torch.tensor(epoch_graph_reg_loss)).item() if epoch_graph_reg_loss else 0.0

            if is_wrong:
                logging.info(f"calculate error in epoch {epoch}")
                break

            if epoch_loss <= self.best_records["loss"]["score"]:
                self.best_records["loss"]["score"] = epoch_loss
                self.best_records["loss"]["state"] = copy.deepcopy(self.model.state_dict())
                self.best_records["loss"]["epoch"] = epoch
                self.save_model(self.best_records['loss'], self.model_save_dir, name='loss')

            logging.info(
                "Epoch {}/{}, all:{:.5f} cls:{:.5f} rec:{:.5f} ctr:{:.5f} greg:{:.5f} [{:.2f}s]; best:{:.5f}"
                .format(
                    epoch,
                    self.epoches,
                    epoch_loss,
                    epoch_cls_loss,
                    epoch_rec_loss,
                    epoch_contrast_loss,
                    epoch_graph_reg_loss,
                    epoch_time_elapsed,
                    self.best_records["loss"]['score'],
                )
            )

            try:
                graph_lambda = torch.sigmoid(self.model.dynamic_graph_learner.graph_lambda).item()
                logging.info(f"  Graph lambda: shared={graph_lambda:.4f}")
            except Exception:
                pass

            try:
                cache_stats = self.model.get_dynamic_graph_cache_stats()
                logging.info(
                    "  Graph cache: refreshes=%d hits=%d hit_rate=%.2f%%"
                    % (
                        cache_stats['refreshes'],
                        cache_stats['hits'],
                        cache_stats['hit_rate'] * 100,
                    )
                )
            except Exception:
                pass

            eval_result = None
            if self._should_eval(epoch, self.eval_interval):
                calibrate_threshold = eval_split == 'val' and self.split_mode == 'formal60_10_30'
                eval_result = self.evaluate(
                    eval_loader,
                    split=eval_split,
                    calibrate_threshold=calibrate_threshold,
                    epoch=epoch,
                    checkpoint_type='latest',
                )

                metric_value = self._metric_value(eval_result['metrics'], self.best_metric_name)
                if metric_value >= self.best_records["f1"]["score"]:
                    self.best_records["f1"]["score"] = metric_value
                    self.best_records["f1"]["state"] = copy.deepcopy(self.model.state_dict())
                    self.best_records["f1"]["epoch"] = epoch
                    self.save_model(self.best_records['f1'], self.model_save_dir, name='f1')

                early_metric_value = self._metric_value(eval_result['metrics'], self.early_stop_metric_name)
                if early_metric_value >= best_early_stop_score:
                    best_early_stop_score = early_metric_value
                    bad_eval_rounds = 0
                else:
                    bad_eval_rounds += 1

                if test_loader is not None and eval_split == 'val':
                    self.evaluate(
                        test_loader,
                        split='test',
                        threshold=eval_result['metrics']['threshold'],
                        calibrate_threshold=False,
                        epoch=epoch,
                        checkpoint_type='latest',
                    )

            if self._should_eval(epoch, self.train_eval_interval):
                train_threshold = None
                if eval_result is not None:
                    train_threshold = eval_result['metrics']['threshold']
                self.evaluate(
                    train_loader,
                    split='train',
                    threshold=train_threshold,
                    calibrate_threshold=False,
                    epoch=epoch,
                    checkpoint_type='latest',
                )

            scheduler.step()

            if self.early_stop_patience > 0 and eval_result is not None and bad_eval_rounds >= self.early_stop_patience:
                logging.info(
                    f"Early stop at epoch: {epoch} ({self.early_stop_metric_name} did not improve for "
                    f"{self.early_stop_patience} eval rounds)"
                )
                break

    def finalize_run(self, eval_loader, test_loader=None, eval_split='test'):
        summary = {
            'run_complete': True,
            'split_mode': self.split_mode,
            'best_metric': self.best_metric_name,
            'score_threshold_mode': self.score_threshold_mode,
            'final_test_f1': None,
            'stages': {},
        }

        logging.info('calculate scores...')
        for stage_name in ['loss', 'f1']:
            logging.info(f'calculate label with {stage_name}...')
            loaded = self.load_model(self.model_save_dir, name=stage_name)
            if not loaded:
                summary['stages'][stage_name] = {
                    'epoch': self.best_records.get(stage_name, {}).get('epoch', -1),
                    'available': False,
                }
                continue

            stage_summary = {
                'available': True,
                'epoch': self.best_records.get(stage_name, {}).get('epoch', -1),
                'selection_score': self.best_records.get(stage_name, {}).get('score'),
            }

            threshold = self._default_threshold()
            eval_result = None
            if eval_loader is not None:
                calibrate_threshold = eval_split == 'val' and self.split_mode == 'formal60_10_30'
                eval_result = self.evaluate(
                    eval_loader,
                    split=eval_split,
                    calibrate_threshold=calibrate_threshold,
                )
                threshold = eval_result['metrics']['threshold']
                stage_summary[f'{eval_split}_metrics'] = eval_result['metrics']

            if test_loader is not None:
                test_result = self.evaluate(
                    test_loader,
                    split='test',
                    threshold=threshold,
                    calibrate_threshold=False,
                )
                stage_summary['test_metrics'] = test_result['metrics']
            elif eval_result is not None and eval_split == 'test':
                stage_summary['test_metrics'] = eval_result['metrics']

            stage_summary['best_threshold'] = threshold
            summary['stages'][stage_name] = stage_summary

        f1_test_metrics = summary['stages'].get('f1', {}).get('test_metrics')
        if f1_test_metrics is not None:
            summary['final_test_f1'] = f1_test_metrics.get('f1')

        with open(self.best_metrics_path, 'w') as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        return summary

    def _fuse_predict_with_reconstruction(self, predict_list, rec_scores):
        cls_probs = predict_list.reshape(-1, predict_list.shape[-1])
        rec_scores = rec_scores.reshape(-1)
        rec_probs = self._reconstruction_energy_to_prob(rec_scores)
        alpha = min(max(self.score_fusion_alpha, 0.0), 1.0)
        fused_anomaly = alpha * cls_probs[:, 1] + (1 - alpha) * rec_probs
        fused_anomaly = fused_anomaly.clamp(0.0, 1.0)
        return torch.stack([1 - fused_anomaly, fused_anomaly], dim=-1)

    def _reconstruction_energy_to_prob(self, rec_scores):
        median = torch.median(rec_scores)
        mad = torch.median(torch.abs(rec_scores - median)).clamp_min(1e-6)
        normalized = (rec_scores - median) / (1.4826 * mad)
        return torch.sigmoid(normalized)
