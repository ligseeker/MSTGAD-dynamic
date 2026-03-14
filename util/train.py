import logging
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief
from torch.cuda.amp import GradScaler, autocast

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
        self.eval_interval = args.get('eval_interval', 5)
        self.rec_down = args['rec_down']
        self.para_low = args['para_low']
        self.True_list = {'normal': 1, 'abnormal': args['abnormal_weight']}

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
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            else:
                logging.info(f"{ckpt_path} not found, skipping load.")

    # Saving modal paras
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
        # === 创新点3: 对比学习参数 ===
        self.contrast_weight = args.get('contrast_weight', 0.1)
        self.contrast_warmup = args.get('contrast_warmup', 5)
        self.contrast_start_epoch = max(0, int(args.get('contrast_start_epoch', self.contrast_warmup)))
        self.graph_update_steps = max(1, int(args.get('graph_update_steps', 4)))
        self.graph_summary_mode = args.get('graph_summary_mode', 'last')
        self.use_amp = bool(args.get('use_amp', True) and self.device.type == 'cuda')

    def _contrast_gamma(self, epoch):
        if self.contrast_weight <= 0 or epoch < self.contrast_start_epoch:
            return 0.0
        if self.contrast_warmup <= 0:
            return self.contrast_weight
        progress = min((epoch - self.contrast_start_epoch + 1) / self.contrast_warmup, 1.0)
        return self.contrast_weight * progress

    def fit(self, train_loader, test_loader, **args):
        optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.learning_change, self.learning_gamma)
        scaler = GradScaler(enabled=self.use_amp)

        best = {"loss":{"score": float("inf"), "state": None, "epoch": 0},
                "f1":{"score": 0, "state": None, "epoch": 0}}

        pre_loss, worse_count, isWrong = float("inf"), 0, False

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
        logging.info(f'  eval_interval={self.eval_interval}, use_amp={self.use_amp}')
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

                    with autocast(enabled=self.use_amp):
                        # 模型现在返回5个值: rec_loss, cls_result, cls_label, contrast_loss, graph_reg_loss
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
                        isWrong = True
                        logging.info("loss is not finite")
                        break

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
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

            if isWrong:
                logging.info("calculate error in epoch {}".format(epoch))
                break

            if epoch_loss <= best["loss"]["score"] or epoch == self.rec_down:
                worse_count = 0
                best["loss"]["score"] = epoch_loss
                best["loss"]["state"] = copy.deepcopy(self.model.state_dict())
                best["loss"]["epoch"] = epoch
            elif epoch_loss <= pre_loss:
                pass
            elif epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break

            pre_loss = epoch_loss
            logging.info(
                "Epoch {}/{}, all:{:.5f} cls:{:.5f} rec:{:.5f} ctr:{:.5f} greg:{:.5f} [{:.2f}s]; best:{:.5f} pat:{}"
                .format(epoch, self.epoches, epoch_loss, epoch_cls_loss, epoch_rec_loss,
                        epoch_contrast_loss, epoch_graph_reg_loss,
                        epoch_time_elapsed, best["loss"]['score'], worse_count))
            
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

            if epoch > self.rec_down and ((epoch + 1) % self.eval_interval == 0 or epoch == self.epoches - 1):
                if hasattr(self.model, 'reset_dynamic_graph_cache'):
                    self.model.reset_dynamic_graph_cache()
                result = self.evaluate(test_loader)
                if float(result['f1']) >= best["f1"]["score"]:
                    best["f1"]["score"] = float(result['f1'])
                    best["f1"]["state"] = copy.deepcopy(self.model.state_dict())
                    best["f1"]["epoch"] = epoch
            scheduler.step()

        logging.info('saving model...')
        self.save_model(best['loss'], self.model_save_dir, name='loss')
        self.save_model(best['f1'], self.model_save_dir, name='f1')

    def evaluate(self, test_loader, isFinall=False):
        self.model.eval()
        if hasattr(self.model, 'reset_dynamic_graph_cache'):
            self.model.reset_dynamic_graph_cache(reset_stats=True)
        with torch.no_grad():
            predict_list, label_list = [], []
            for batch_input in tqdm(test_loader):
                    batch_input = self.input2device(batch_input,self.use_gpu)
                    raw_result, _ = self.model(batch_input, evaluate=True)

                    predict_list.append(raw_result)
                    label_list.append(batch_input['groundtruth_real'])

            predict_list = torch.concat(predict_list, dim=0).cpu()
            label_list = torch.concat(label_list, dim=0).cpu()

            info, result = util.calc_index(predict_list, label_list)

            if isFinall:
                return info
            else:
                return result
