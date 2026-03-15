import logging
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief
from sklearn.metrics import f1_score

from tqdm import tqdm
import util.util as util

class Base(nn.Module):
    def __init__(self, model, **args):
        super(Base, self).__init__()

        self.model = model
        self.use_gpu = args['gpu']
        self.device = torch.device('cuda' if args['gpu'] and torch.cuda.is_available() else 'cpu')
        # Training
        self.epoches = args['epochs']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']
        self.patience = args['patience']  # > 0: use early stop
        self.model_save_dir = args['result_dir']
        self.learning_change = args['learning_change']
        self.learning_gamma = args['learning_gamma']
        self.eval_interval = max(1, int(args.get('eval_interval', 5)))
        self.train_eval_interval = max(0, int(args.get('train_eval_interval', 1)))
        self.monitor_metric = str(args.get('monitor_metric', 'f1')).lower()
        if self.monitor_metric not in {'f1', 'auc', 'ap', 'pr', 'rc'}:
            logging.info(f"Unknown monitor_metric={self.monitor_metric}, fallback to f1")
            self.monitor_metric = 'f1'
        self.monitor_patience = max(1, int(args.get('monitor_patience', 6)))
        self.monitor_warmup_epochs = max(0, int(args.get('monitor_warmup_epochs', 20)))
        self.monitor_min_delta = float(args.get('monitor_min_delta', 0.002))
        self.threshold_search = bool(args.get('threshold_search', True))
        self.threshold_search_on = str(args.get('threshold_search_on', 'train')).lower()
        if self.threshold_search_on not in {'train', 'test'}:
            logging.info(f"Unknown threshold_search_on={self.threshold_search_on}, fallback to train")
            self.threshold_search_on = 'train'
        self.threshold_grid_step = max(1e-4, float(args.get('threshold_grid_step', 0.01)))
        self.threshold_ema = min(max(float(args.get('threshold_ema', 0.8)), 0.0), 0.999)
        self.plateau_factor = min(max(float(args.get('plateau_factor', 0.5)), 0.1), 0.99)
        self.plateau_patience = max(1, int(args.get('plateau_patience', 4)))
        self.rec_down = args['rec_down']
        self.para_low = args['para_low']
        self.True_list = {'normal': 1, 'abnormal': args['abnormal_weight']}
        self._loaded_eval_threshold = None

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

    # Model init
    def init_weight(self):
        for p in self.model.parameters():
            if p.dim() > 1 and all(s > 0 for s in p.shape):
                nn.init.xavier_uniform_(p)

    #  Put Data into GPU/CPU
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

    # Loading modal paras
    def load_model(self, model_save_file="", name='loss'):
        if model_save_file == ' ':
            logging.info(f'No {self.model.name} statue file')
        else:
            logging.info(f'{self.model.name} on {model_save_file} loading...')
            ckpt_path = os.path.join(model_save_file, f"{self.model.name}_{name}_stage.ckpt")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    meta = checkpoint.get('meta', {})
                else:
                    state_dict = checkpoint
                    meta = {}
                self.model.load_state_dict(state_dict)

                threshold = meta.get('threshold', None)
                if threshold is not None:
                    try:
                        threshold = float(threshold)
                        self._loaded_eval_threshold = threshold
                        if hasattr(self, 'eval_threshold'):
                            self.eval_threshold = threshold
                        if hasattr(self, '_threshold_initialized'):
                            self._threshold_initialized = True
                        logging.info(f'loaded threshold from {name} ckpt: {threshold:.4f}')
                    except (TypeError, ValueError):
                        pass
            else:
                logging.info(f"{ckpt_path} not found, skipping load.")

    # Saving modal paras
    def save_model(self, best_dict, model_save_dir="", name='loss'):
        os.makedirs(model_save_dir, exist_ok=True)
        file_status = os.path.join(model_save_dir, f"{self.model.name}_{name}_stage.ckpt")
        if best_dict['state'] is None:
            logging.info(f'No {self.model.name} - {name} statue file')
        else: 
            threshold = best_dict.get('threshold', None)
            if threshold is None:
                logging.info(
                    f'{self.model.name} - {name}  best score:{best_dict["score"]} at epoch {best_dict["epoch"]}'
                )
            else:
                logging.info(
                    f'{self.model.name} - {name}  best score:{best_dict["score"]} '
                    f'at epoch {best_dict["epoch"]} threshold:{threshold:.4f}'
                )
            payload = {
                'model_state_dict': best_dict['state'],
                'meta': {
                    'score': float(best_dict.get('score', 0.0)),
                    'epoch': int(best_dict.get('epoch', 0)),
                    'threshold': threshold,
                    'name': name,
                }
            }
            torch.save(payload, file_status)

class MY(Base):
    def __init__(self, model, **args):
        super().__init__(model, **args)
        # === 创新点3: 对比学习参数 ===
        self.contrast_weight = args.get('contrast_weight', 0.1)
        self.contrast_warmup = args.get('contrast_warmup', 5)
        self.contrast_start_epoch = max(0, int(args.get('contrast_start_epoch', self.contrast_warmup)))
        self.graph_update_steps = max(1, int(args.get('graph_update_steps', 4)))
        self.graph_summary_mode = args.get('graph_summary_mode', 'last')
        self.contrast_summary_mode = args.get('contrast_summary_mode', 'last')
        self.score_fusion_alpha = float(args.get('score_fusion_alpha', 1.0))
        self.eval_threshold = float(self._loaded_eval_threshold) if self._loaded_eval_threshold is not None else 0.5
        self._threshold_initialized = self._loaded_eval_threshold is not None

    def _contrast_gamma(self, epoch):
        if self.contrast_weight <= 0 or epoch < self.contrast_start_epoch:
            return 0.0
        if self.contrast_warmup <= 0:
            return self.contrast_weight
        progress = min((epoch - self.contrast_start_epoch + 1) / self.contrast_warmup, 1.0)
        return self.contrast_weight * progress

    def fit(self, train_loader, test_loader, **args):
        optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.plateau_factor,
            patience=self.plateau_patience,
            min_lr=1e-5,
        )

        best = {"loss":{"score": float("inf"), "state": None, "epoch": 0, "threshold": None},
                "f1":{"score": float("-inf"), "state": None, "epoch": 0, "threshold": None}}
        monitor_best = float("-inf")
        monitor_bad_count = 0
        monitor_best_epoch = -1
        monitor_tracking_started = False

        is_wrong = False

        label_weight = torch.tensor(
            np.array(list(self.True_list.values())), dtype=torch.float, device=self.device)
        losser = nn.BCEWithLogitsLoss(reduction='mean', weight=label_weight)
        logging.info('optimizer : using AdaBelief')
        logging.info(f'Innovations: Dynamic Graph + Gated Fusion + Contrastive Learning')
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
            f'  eval_interval={self.eval_interval}, train_eval_interval={self.train_eval_interval}, precision=float32'
        )
        logging.info(
            f'  monitor={self.monitor_metric}, monitor_patience={self.monitor_patience}, '
            f'warmup={self.monitor_warmup_epochs}, min_delta={self.monitor_min_delta}'
        )
        logging.info(
            f'  threshold_search={self.threshold_search}, threshold_search_on={self.threshold_search_on}, '
            f'threshold_grid_step={self.threshold_grid_step}, threshold_ema={self.threshold_ema}'
        )
        logging.info(
            f'  scheduler=ReduceLROnPlateau(mode=max, factor={self.plateau_factor}, '
            f'patience={self.plateau_patience}, min_lr=1e-5)'
        )
        global_step = 0

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
                f'{epoch}/{self.epoches} starting... lr: {lr} para:{para} gamma(contrast):{gamma:.4f}')
            self.model.train()
            epoch_cls_loss, epoch_rec_loss, epoch_loss = [], [], []
            epoch_contrast_loss, epoch_graph_reg_loss = [], []
            epoch_time_start = time.time()
            with tqdm(train_loader) as tbar:
                for batch_input in tbar:
                    batch_input = self.input2device(batch_input, self.use_gpu)
                    optimizer.zero_grad()

                    # 真实 MSDS 上 AMP 会触发非有限 loss，这里固定使用 float32 训练。
                    raw_loss, cls_result, cls_label, contrast_loss, graph_reg_loss = self.model(
                        batch_input,
                        global_step=global_step,
                        compute_contrast=compute_contrast
                    )

                    rec_loss = sum(raw_loss)
                    if cls_result.shape[0] == 0:
                        cls_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    else:
                        cls_loss = losser(cls_result, cls_label)

                    # === 综合损失 = 分类损失 + 重构损失 + 对比损失 + 图正则化 ===
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
                        loss=f'{loss.item():.6f}', cls=f'{cls_loss.item():.6f}',
                        rec=f'{rec_loss.item():.6f}', ctr=f'{contrast_loss.item():.4f}')

            # show the result about this epoch
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = torch.mean(torch.tensor(epoch_loss)).item() if epoch_loss else float("inf")
            epoch_cls_loss = torch.mean(torch.tensor(epoch_cls_loss)).item() if epoch_cls_loss else 0.0
            epoch_rec_loss = torch.mean(torch.tensor(epoch_rec_loss)).item() if epoch_rec_loss else 0.0
            epoch_contrast_loss = torch.mean(torch.tensor(epoch_contrast_loss)).item() if epoch_contrast_loss else 0.0
            epoch_graph_reg_loss = torch.mean(torch.tensor(epoch_graph_reg_loss)).item() if epoch_graph_reg_loss else 0.0

            if is_wrong:
                logging.info("calculate error in epoch {}".format(epoch))
                break

            if epoch_loss <= best["loss"]["score"] or epoch == self.rec_down:
                best["loss"]["score"] = epoch_loss
                best["loss"]["state"] = copy.deepcopy(self.model.state_dict())
                best["loss"]["epoch"] = epoch
                best["loss"]["threshold"] = float(self.eval_threshold)
            logging.info(
                "Epoch {}/{}, all:{:.5f} cls:{:.5f} rec:{:.5f} ctr:{:.5f} greg:{:.5f} [{:.2f}s]; "
                "best_loss:{:.5f} monitor_pat:{}"
                .format(epoch, self.epoches, epoch_loss, epoch_cls_loss, epoch_rec_loss,
                        epoch_contrast_loss, epoch_graph_reg_loss,
                        epoch_time_elapsed, best["loss"]['score'], monitor_bad_count))
            
            try:
                graph_lambda = torch.sigmoid(self.model.dynamic_graph_learner.graph_lambda).item()
                logging.info(f"  Graph λ: shared={graph_lambda:.4f}")
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

            should_eval_train = self.train_eval_interval > 0 and (
                (epoch + 1) % self.train_eval_interval == 0 or epoch == self.epoches - 1
            )
            should_eval_test = epoch > self.rec_down and (
                (epoch + 1) % self.eval_interval == 0 or epoch == self.epoches - 1
            )

            calibrated_threshold = self.eval_threshold
            train_predict, train_label = None, None
            need_train_for_threshold = (
                should_eval_test and self.threshold_search and self.threshold_search_on == 'train'
            )
            if should_eval_train or need_train_for_threshold:
                train_predict, train_label = self._collect_eval_outputs(train_loader)

            if (
                should_eval_test
                and self.threshold_search
                and self.threshold_search_on == 'train'
                and train_predict is not None
            ):
                raw_threshold, best_train_f1 = self._search_best_threshold(train_predict, train_label)
                calibrated_threshold = self._update_eval_threshold(raw_threshold)
                logging.info(
                    "  Threshold calibration(train): raw={:.4f} best_train_f1={:.4f} ema={:.4f} "
                    "(range=[0.05,0.95], step={:.4f})".format(
                        raw_threshold, best_train_f1, calibrated_threshold, self.threshold_grid_step
                    )
                )

            if should_eval_train and train_predict is not None:
                _, train_result = util.calc_index(
                    train_predict, train_label, threshold=calibrated_threshold, log=False
                )
                logging.info(
                    "[train] pr:{pr:.4f}  rc:{rc:.4f}  auc:{auc:.4f} ap:{ap:.4f} "
                    "f1:{f1:.4f} thr:{threshold:.4f}".format(**train_result)
                )

            if should_eval_test:
                test_predict, test_label = self._collect_eval_outputs(test_loader)
                if self.threshold_search and self.threshold_search_on == 'test':
                    raw_threshold, best_test_f1 = self._search_best_threshold(test_predict, test_label)
                    calibrated_threshold = self._update_eval_threshold(raw_threshold)
                    logging.info(
                        "  Threshold calibration(test): raw={:.4f} best_test_f1={:.4f} ema={:.4f} "
                        "(range=[0.05,0.95], step={:.4f})".format(
                            raw_threshold, best_test_f1, calibrated_threshold, self.threshold_grid_step
                        )
                    )
                _, result = util.calc_index(
                    test_predict, test_label, threshold=calibrated_threshold, log=False
                )
                logging.info(
                    "[test] pr:{pr:.4f}  rc:{rc:.4f}  auc:{auc:.4f} ap:{ap:.4f} "
                    "f1:{f1:.4f} thr:{threshold:.4f}".format(**result)
                )

                current_f1 = float(result['f1'])
                monitor_score = float(result.get(self.monitor_metric, current_f1))
                if current_f1 >= best["f1"]["score"]:
                    best["f1"]["score"] = current_f1
                    best["f1"]["state"] = copy.deepcopy(self.model.state_dict())
                    best["f1"]["epoch"] = epoch
                    best["f1"]["threshold"] = float(calibrated_threshold)

                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(monitor_score)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr:
                    logging.info(
                        "  LR reduced by plateau scheduler: %.8f -> %.8f (monitor %.4f)"
                        % (prev_lr, new_lr, monitor_score)
                    )

                if epoch >= self.monitor_warmup_epochs:
                    # Warmup 后重置监控基线，避免被 warmup 阶段的偶发高点提前触发 early stop。
                    if not monitor_tracking_started:
                        monitor_tracking_started = True
                        monitor_best = monitor_score
                        monitor_bad_count = 0
                        monitor_best_epoch = epoch
                        logging.info(
                            "  monitor tracking starts at epoch %d: baseline=%.4f (%s)"
                            % (epoch, monitor_best, self.monitor_metric)
                        )
                    elif monitor_score > monitor_best + self.monitor_min_delta:
                        monitor_best = monitor_score
                        monitor_bad_count = 0
                        monitor_best_epoch = epoch
                    else:
                        monitor_bad_count += 1
                        logging.info(
                            "  monitor plateau: current=%.4f best=%.4f at epoch %d "
                            "(bad_count=%d/%d, min_delta=%.4f)"
                            % (
                                monitor_score,
                                monitor_best,
                                monitor_best_epoch,
                                monitor_bad_count,
                                self.monitor_patience,
                                self.monitor_min_delta,
                            )
                        )
                        if monitor_bad_count >= self.monitor_patience:
                            logging.info(
                                "Early stop at epoch %d by monitor(%s): no improvement for %d eval rounds"
                                % (epoch, self.monitor_metric, self.monitor_patience)
                            )
                            break
                elif monitor_score > monitor_best:
                    monitor_best = monitor_score
                    monitor_best_epoch = epoch

        logging.info('saving model...')
        if best['loss']['state'] is None:
            best['loss']['state'] = copy.deepcopy(self.model.state_dict())
            best['loss']['score'] = float("inf")
            best['loss']['epoch'] = self.epoches - 1
            best['loss']['threshold'] = float(self.eval_threshold)
        if best['f1']['state'] is None:
            best['f1']['state'] = copy.deepcopy(self.model.state_dict())
            best['f1']['score'] = monitor_best if monitor_best > float("-inf") else 0.0
            best['f1']['epoch'] = monitor_best_epoch if monitor_best_epoch >= 0 else self.epoches - 1
            best['f1']['threshold'] = float(self.eval_threshold)
        logging.info(f'final calibrated threshold: {self.eval_threshold:.4f}')
        self.save_model(best['loss'], self.model_save_dir, name='loss')
        self.save_model(best['f1'], self.model_save_dir, name='f1')

    def _collect_eval_outputs(self, data_loader):
        self.model.eval()
        if hasattr(self.model, 'reset_dynamic_graph_cache'):
            self.model.reset_dynamic_graph_cache(reset_stats=True)
        with torch.no_grad():
            predict_list, label_list, rec_score_list = [], [], []
            for batch_input in tqdm(data_loader):
                    batch_input = self.input2device(batch_input,self.use_gpu)
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
            return predict_list, label_list

    def _search_best_threshold(self, predict_list, label_list):
        if predict_list.shape[-1] != 2 or label_list.shape[-1] != 2:
            return self.eval_threshold, 0.0
        y_true = torch.argmax(label_list.reshape(-1, label_list.shape[-1]), dim=-1).numpy()
        y_prob = predict_list.reshape(-1, predict_list.shape[-1])[:, 1].numpy()

        low, high = 0.05, 0.95
        thresholds = np.arange(low, high + 1e-8, self.threshold_grid_step)
        if thresholds.size == 0:
            return self.eval_threshold, 0.0

        best_threshold = float(self.eval_threshold)
        best_f1 = float("-inf")
        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(np.int64)
            score = float(f1_score(y_true, y_pred, average="binary", zero_division=1))
            if score > best_f1 + 1e-12 or (abs(score - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold, best_f1

    def _update_eval_threshold(self, raw_threshold):
        clipped = float(np.clip(raw_threshold, 0.05, 0.95))
        if self._threshold_initialized:
            updated = self.threshold_ema * self.eval_threshold + (1 - self.threshold_ema) * clipped
        else:
            updated = clipped
            self._threshold_initialized = True
        self.eval_threshold = float(np.clip(updated, 0.05, 0.95))
        return self.eval_threshold

    def evaluate(self, test_loader, isFinall=False, threshold=None):
        predict_list, label_list = self._collect_eval_outputs(test_loader)
        eval_threshold = self.eval_threshold if threshold is None else float(threshold)
        info, result = util.calc_index(predict_list, label_list, threshold=eval_threshold)

        if isFinall:
            return info
        else:
            return result

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
