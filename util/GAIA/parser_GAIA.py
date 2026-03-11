import argparse

parser = argparse.ArgumentParser(description='MSTGAD on GAIA Dataset')
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
# GAIA: 76 metric features per node (after intersection + variance filter)
parser.add_argument("--feature_node", default=16, type=int,
                    help='embedded node feature dimension (larger due to 76 raw features)')
parser.add_argument("--feature_edge", default=4, type=int,
                    help='embedded edge feature dimension')
parser.add_argument("--feature_log", default=8, type=int,
                    help='embedded log feature dimension')
parser.add_argument("--raw_node", default=0, type=int,
                    help='raw metric features per node (0=auto-detect from data)')
parser.add_argument("--raw_edge", default=0, type=int,
                    help='raw trace feature types (0=auto-detect from data)')
parser.add_argument("--log_len", default=0, type=int,
                    help='the log template amount (0=auto-detect from data)')
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

# dataset setting
parser.add_argument("--batch_size", default=32, type=int,
                    help='batch size (reduced from 50 due to larger data)')
parser.add_argument("--window", default=10, type=int,
                    help='size of sliding window (10 x 30s = 5 minutes)')
parser.add_argument("--step", default=1, type=int,
                    help='sliding window stride')
parser.add_argument("--max_timesteps", default=0, type=int,
                    help='max timesteps to load (0=all, e.g. 5000 for quick test)')
parser.add_argument("--num_nodes", default=10, type=int,
                    help='the number of service nodes in GAIA')

# path setting
parser.add_argument("--data_path", default='./data/GAIA-pre',
                    type=str, help='the path of preprocessed data')
parser.add_argument("--dataset_path", default="./data/GAIA-save",
                    type=str, help='the path of saving windowed data')
parser.add_argument("--result_dir", default="./result",
                    type=str, help='the path of result and log')

parser.add_argument("--main_model", default='MSTGAD', type=str,
                    help='switch the model that will run')
parser.add_argument("--evaluate", default=False,
                    type=lambda x: x.lower() == "true", help='Evaluate the exist model')
parser.add_argument("--model_path", default=None,
                    type=str, help='the path of exist model')

args = vars(parser.parse_args([]))
