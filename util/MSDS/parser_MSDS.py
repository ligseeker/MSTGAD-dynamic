import argparse

parser = argparse.ArgumentParser(description='MutliModel Time-Series Anomaly Detection')
parser.add_argument("--random_seed", default=42,
                    type=int, help='the random seed')

# training setting
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epochs", default=300, type=int,
                    help='the number of training epochs')
parser.add_argument("--patience", default=15, type=float,
                    help='the number of epoch that loss is uping')
parser.add_argument("--learning_rate", default=1e-3,
                    type=float, help='the data number at one epoch')
parser.add_argument("--weight_decay", default=5e-4, type=float,
                    help='the one of optimzier parameters which prevent overfitting ')
parser.add_argument("--learning_change", default=100, type=int,
                    help='the epoch number that change learning rate')
parser.add_argument("--learning_gamma", default=0.9, type=float,
                    help='the weight that change learning rate')
parser.add_argument("--eval_interval", default=5, type=int,
                    help='evaluate every N epochs after warm-up')
parser.add_argument("--label_weight", default=1e-2, type=float,
                    help='the unkown weight in reconstruction loss')
parser.add_argument("--label_percent", default=0.5, type=float,
                    help='the proportion of labeled data') 
parser.add_argument("--abnormal_weight", default=96, type=int,
                    help='the abnormal weight in classfication loss')
parser.add_argument("--rec_down", default=1, type=int,
                    help='the number that changes reconstruction loss weight')
parser.add_argument("--para_low", default=1e-2, type=float,
                    help='the min weight of rec loss')

# model setting
parser.add_argument("--feature_node", default=4, type=int,
                    help='the pod kpi data number at one epoch')
parser.add_argument("--feature_edge", default=4, type=int,
                    help='the edge data number at one epoch')
parser.add_argument("--feature_log", default=16, type=int,
                    help='the log data number at one epoch')
parser.add_argument("--raw_node", default=3, type=int,
                    help='the raw pod kpi data number at one epoch')
parser.add_argument("--raw_edge", default=7, type=int,
                    help='the raw edge kpi data number at one epoch')
parser.add_argument("--log_len", default=256, type=int,
                    help='the log template amount')
parser.add_argument("--num_heads_edge", default=4, type=int,
                    help='the number of multiattention heads about trace')
parser.add_argument("--num_heads_node", default=4, type=int,
                    help='the number of multiattention heads about metric')
parser.add_argument("--num_heads_log", default=4, type=int,
                    help='the number of multiattention heads about log')
parser.add_argument("--num_heads_n2e", default=4, type=int,
                    help='the number of multiattention heads about node')
parser.add_argument("--num_heads_e2n", default=2, type=int,
                    help='the number of multiattention heads about edge')
parser.add_argument("--num_layer", default=2, type=int,
                    help='the number of model layers')
parser.add_argument("--dropout", default=0.2, type=float)

# === Innovation parameters ===
# Dynamic Graph
parser.add_argument("--graph_hidden", default=16, type=int,
                    help='hidden dim for dynamic graph learner')
parser.add_argument("--graph_sparse_weight", default=1e-3, type=float,
                    help='graph regularization weight (sparsity + KL)')
parser.add_argument("--graph_update_steps", default=4, type=int,
                    help='refresh dynamic graph every N training steps')
parser.add_argument("--graph_summary_mode", default='last', type=str,
                    choices=['last', 'mean'],
                    help='feature summary mode for dynamic graph learner')
# Contrastive Learning
parser.add_argument("--contrast_weight", default=0.1, type=float,
                    help='contrastive loss weight')
parser.add_argument("--contrast_temp", default=0.1, type=float,
                    help='InfoNCE temperature')
parser.add_argument("--contrast_proj_dim", default=32, type=int,
                    help='projection dim for contrastive head')
parser.add_argument("--contrast_start_epoch", default=5, type=int,
                    help='delay contrastive branch until this epoch')
parser.add_argument("--contrast_warmup", default=5, type=int,
                    help='warm-up epochs for contrastive loss')
parser.add_argument("--use_amp", default=True, type=lambda x: x.lower() == "true",
                    help='use automatic mixed precision on CUDA')


# dataset setting
parser.add_argument("--batch_size", default=50, type=int,
                    help='the data number at one epoch')
parser.add_argument("--window", default=10, type=int,
                    help='size of sliding window')
parser.add_argument("--step", default=1, type=int,
                    help='sliding window stride')
parser.add_argument("--num_nodes", default=5, type=int,
                    help='the number of node in graph')
parser.add_argument("--num_workers", default=4, type=int,
                    help='num workers for dataloader')
parser.add_argument("--pin_memory", default=True, type=lambda x: x.lower() == "true",
                    help='pin host memory for faster device transfer')
parser.add_argument("--persistent_workers", default=True, type=lambda x: x.lower() == "true",
                    help='keep dataloader workers alive between epochs')

# path setting
parser.add_argument("--data_path", default='./data/MSDS-pre',
                    type=str, help='the path of raw data')
parser.add_argument("--dataset_path", default="./data/MSDS-save",
                    type=str, help='the path of saving data')
parser.add_argument("--result_dir", default="./result",
                    type=str, help='the path of result and log')

parser.add_argument("--main_model", default='MSTGAD', type=str,
                    help='switch the model that will run')
parser.add_argument("--evaluate", default=False, 
                    type=lambda x: x.lower() == "true", help='Evaluate the exist model')
parser.add_argument("--model_path", default=None,
                    type=str, help=' the path of exist model')

args = vars(parser.parse_args())
