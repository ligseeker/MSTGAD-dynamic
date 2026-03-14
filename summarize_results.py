import json
import os

import pandas as pd


def parse_folder_name(folder_name):
    parts = folder_name.split('-')
    try:
        timestamp = parts[-1]
        hash_val = parts[-2]
        if len(parts) >= 4 and parts[-3] == 'save':
            dataset = parts[-4]
            method = "-".join(parts[:-4])
        else:
            dataset = parts[-3]
            method = "-".join(parts[:-3])
        return method, dataset, hash_val, timestamp
    except Exception:
        return folder_name, "Unknown", "Unknown", "Unknown"


def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def flatten_stage(stage):
    row = {
        'epoch': stage.get('epoch'),
        'threshold': stage.get('best_threshold'),
    }
    for split in ['val', 'test']:
        metrics = stage.get(f'{split}_metrics', {})
        for key in ['pr', 'rc', 'auc', 'ap', 'f1']:
            row[f'{split}_{key}'] = metrics.get(key)
    return row


def build_row(result_dir, folder_name):
    best_metrics_path = os.path.join(result_dir, folder_name, 'best_metrics.json')
    params_path = os.path.join(result_dir, folder_name, 'params.json')
    if not os.path.exists(best_metrics_path) or not os.path.exists(params_path):
        return None

    best_metrics = read_json(best_metrics_path)
    if not best_metrics.get('run_complete', False):
        return None

    params = read_json(params_path)
    method, dataset, hash_val, timestamp = parse_folder_name(folder_name)

    loss_stage = flatten_stage(best_metrics.get('stages', {}).get('loss', {}))
    f1_stage = flatten_stage(best_metrics.get('stages', {}).get('f1', {}))

    return {
        'Experiment_ID': folder_name,
        'Method': method,
        'Dataset': dataset,
        'Hash': hash_val,
        'Timestamp': timestamp,
        'Split_Mode': best_metrics.get('split_mode', params.get('split_mode', 'baseline70_30')),
        'Best_Metric': best_metrics.get('best_metric', params.get('best_metric', 'val_f1')),
        'Eval_Interval': params.get('eval_interval'),
        'Train_Eval_Interval': params.get('train_eval_interval'),
        'Best_Epoch': f1_stage.get('epoch'),
        'Best_Threshold': f1_stage.get('threshold'),
        'Final_Test_F1': best_metrics.get('final_test_f1'),
        'Loss_Epoch': loss_stage.get('epoch'),
        'Loss_Threshold': loss_stage.get('threshold'),
        'Loss_Val_PR': loss_stage.get('val_pr'),
        'Loss_Val_RC': loss_stage.get('val_rc'),
        'Loss_Val_AUC': loss_stage.get('val_auc'),
        'Loss_Val_AP': loss_stage.get('val_ap'),
        'Loss_Val_F1': loss_stage.get('val_f1'),
        'Loss_Test_PR': loss_stage.get('test_pr'),
        'Loss_Test_RC': loss_stage.get('test_rc'),
        'Loss_Test_AUC': loss_stage.get('test_auc'),
        'Loss_Test_AP': loss_stage.get('test_ap'),
        'Loss_Test_F1': loss_stage.get('test_f1'),
        'F1_Epoch': f1_stage.get('epoch'),
        'F1_Threshold': f1_stage.get('threshold'),
        'F1_Val_PR': f1_stage.get('val_pr'),
        'F1_Val_RC': f1_stage.get('val_rc'),
        'F1_Val_AUC': f1_stage.get('val_auc'),
        'F1_Val_AP': f1_stage.get('val_ap'),
        'F1_Val_F1': f1_stage.get('val_f1'),
        'F1_Test_PR': f1_stage.get('test_pr'),
        'F1_Test_RC': f1_stage.get('test_rc'),
        'F1_Test_AUC': f1_stage.get('test_auc'),
        'F1_Test_AP': f1_stage.get('test_ap'),
        'F1_Test_F1': f1_stage.get('test_f1'),
    }


def summarize_results(result_dir='result', output_csv='result/summary_results.csv'):
    rows = []
    for folder_name in sorted(os.listdir(result_dir)):
        folder_path = os.path.join(result_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        row = build_row(result_dir, folder_name)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No finished experiments with best_metrics.json found.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(by=['Timestamp', 'Experiment_ID'], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully summarized {len(rows)} experiments to {output_csv}")


if __name__ == "__main__":
    summarize_results()
