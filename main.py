import util.util as util
import util.train as train
import util.MSDS.data_MSDS as data_loads
from util.MSDS.parser_MSDS import *

from torch.utils.data import DataLoader
import warnings
import logging
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/code')
warnings.filterwarnings("ignore")

util.seed_everything(args['random_seed'])

def build_split_ranges(dataset_len, split_mode, val_ratio):
    baseline_split = int(dataset_len * 0.7)
    if split_mode == 'baseline70_30':
        return {
            'train': (0, baseline_split),
            'eval': (baseline_split, dataset_len),
            'test': (baseline_split, dataset_len),
            'eval_split': 'test',
        }

    if not 0 < val_ratio < 0.7:
        raise ValueError(f"val_ratio must be in (0, 0.7), got {val_ratio}")

    val_start = int(dataset_len * (0.7 - val_ratio))
    return {
        'train': (0, val_start),
        'eval': (val_start, baseline_split),
        'test': (baseline_split, dataset_len),
        'eval_split': 'val',
    }


def build_loader(dataset, batch_size, shuffle, args):
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': args['num_workers'],
        'pin_memory': bool(args['pin_memory'] and args['gpu']),
        'drop_last': True,
    }
    if args['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = args['persistent_workers']
    return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)


def write_summary_log(summary, args):
    with open('./result.log', 'a+') as file:
        file.writelines(
            f"\n {args['main_model']}-{args['hash_id']} --weight_decay:{args['weight_decay']}   "
            f"--learning_change:{args['learning_change']} --split_mode:{args['split_mode']}\n"
        )
        for stage_name in ['loss', 'f1']:
            stage = summary.get('stages', {}).get(stage_name, {})
            test_metrics = stage.get('test_metrics')
            if test_metrics is None:
                file.writelines(f"{stage_name}   unavailable\n")
                continue
            file.writelines(f"{stage_name}   {util.format_binary_metrics(test_metrics)}\n")


if __name__ == '__main__':
    if args['evaluate']:
        dict_json = util.read_params(args)
        for key in dict_json.keys():
            args[key] = args[key] if key in ['model_path', 'evaluate', 'result_dir', 'data_path', 'dataset_path'] else dict_json[key]
        args['result_dir'] = args['model_path']
        args.setdefault('hash_id', os.path.basename(args['model_path']).split('-')[-2] if '-' in os.path.basename(args['model_path']) else 'eval')
    else:
        args['hash_id'], args['result_dir'] = util.dump_params(args)
        util.json_pretty_dump(args, os.path.join(args['result_dir'], "params.json"))
        args['model_path'] = args['result_dir']

    logging.info("---- Model: ----" + args['main_model'] +"-" + args['hash_id'] + "----" + f"train : {not args['evaluate']}"\
        + "----" + f"evaluate : {args['evaluate']}")
    logging.info(f"split_mode={args['split_mode']} eval_interval={args['eval_interval']} train_eval_interval={args['train_eval_interval']}")

    # dealing & loading data
    processed = data_loads.Process(**args)
    split_info = build_split_ranges(len(processed.dataset), args['split_mode'], args['val_ratio'])
    train_start, train_end = split_info['train']
    eval_start, eval_end = split_info['eval']
    test_start, test_end = split_info['test']
    eval_split = split_info['eval_split']

    train_dataset = processed.dataset[train_start:train_end]
    eval_dataset = processed.dataset[eval_start:eval_end]
    test_dataset = processed.dataset[test_start:test_end]

    logging.info(
        f"dataset split -> train:{len(train_dataset)} {eval_split}:{len(eval_dataset)} test:{len(test_dataset)}"
    )

    train_dl = build_loader(train_dataset, args['batch_size'], True, args)
    eval_dl = build_loader(eval_dataset, args['batch_size'], False, args)
    test_dl = build_loader(test_dataset, args['batch_size'], False, args)

    # declear model and train
    import src.model as model
    models = model.MyModel(processed.graph, **args)
    sys = train.MY(models, **args)  

    #Training
    if not args['evaluate']:
        sys.fit(
            train_loader=train_dl,
            eval_loader=eval_dl,
            eval_split=eval_split,
            test_loader=test_dl if args['split_mode'] == 'formal60_10_30' else None,
        )

    summary = sys.finalize_run(
        eval_loader=eval_dl,
        test_loader=None if args['split_mode'] == 'baseline70_30' else test_dl,
        eval_split=eval_split,
    )
    write_summary_log(summary, args)
    logging.info("^^^^^^ Current Model: ----" + args['main_model'] + "-" * 4 + args['hash_id'] + " ^^^^^")

