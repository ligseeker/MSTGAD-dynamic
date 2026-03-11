import os
import re
import csv
import pandas as pd

def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取 Method, Dataset, Hash, Timestamp
    格式示例: MSTGAD-MSDS-save-4422be38-1766594176
    """
    parts = folder_name.split('-')
    
    # 从后往前提取固定模式
    try:
        # 假设最后是时间戳，倒数第二是Hash
        timestamp = parts[-1]
        hash_val = parts[-2]
        
        # 检查是否有 'save'，如果是 MSTGAD-MSDS-save-...
        if len(parts) >= 4 and parts[-3] == 'save':
            dataset = parts[-4]
            method = "-".join(parts[:-4])
        # 如果是 MSTGAD-MSDS-...
        else:
            dataset = parts[-3]
            method = "-".join(parts[:-3])
            
        return method, dataset, hash_val, timestamp
    except:
        return folder_name, "Unknown", "Unknown", "Unknown"

def get_metrics_from_lines(lines, mode_label="calculate label with f1"):
    """
    从日志行中提取指定模式下的指标
    """
    start_parsing = False
    
    # 正序遍历这些行
    for line in lines:
        if mode_label in line:
            start_parsing = True
            continue
        
        if start_parsing and "INFO pr:" in line:
            # 找到指标行，进行解析
            # 格式: ... INFO pr:0.8876  rc:0.9091  auc:0.9710 ap:0.9375 f1: 0.8982 ...
            match = re.search(r'pr:([\d\.]+)\s+rc:([\d\.]+)\s+auc:([\d\.]+)\s+ap:([\d\.]+)\s+f1:\s*([\d\.]+)', line)
            if match:
                return match.groups() # pr, rc, auc, ap, f1
    return None

def summarize_results(result_dir='result', output_csv='result/summary_results.csv'):
    
    # 1. 读取现有的 CSV 以进行过滤 (如果存在)
    existing_folders = set()
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            if 'Experiment_ID' in df.columns:
                existing_folders = set(df['Experiment_ID'].tolist())
        except Exception as e:
            print(f"Warning: Could not read existing csv: {e}")

    new_results = []
    
    # 2. 遍历 result 目录
    for root, dirs, files in os.walk(result_dir):
        folder_name = os.path.basename(root)
        
        # 如果 CSV 中已有，则跳过
        if folder_name in existing_folders:
            continue

        if 'running.log' in files:
            log_path = os.path.join(root, 'running.log')
            
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 读取所有行取最后 12 行
                    all_lines = f.readlines()
                    if len(all_lines) > 12:
                        lines = all_lines[-12:] 
                    else:
                        lines = all_lines
                
                if not lines:
                    continue

                # 3. 检查最后一行是否有完成标识
                last_line = lines[-1].strip()
                if "^^^^^^ Current Model:" not in last_line:
                    continue
                
                # 4. 解析文件夹信息
                method, dataset, hash_val, timestamp = parse_folder_name(folder_name)
                
                # 5. 提取 Best Loss 和 Best F1 结果
                loss_metrics = get_metrics_from_lines(lines, mode_label="calculate label with loss")
                f1_metrics = get_metrics_from_lines(lines, mode_label="calculate label with f1")
                
                if loss_metrics and f1_metrics:
                    loss_pr, loss_rc, loss_auc, loss_ap, loss_f1 = loss_metrics
                    f1_pr, f1_rc, f1_auc, f1_ap, f1_f1 = f1_metrics
                    
                    print(f"Parsed: {folder_name} -> Loss F1: {loss_f1}, Best F1: {f1_f1}")
                    
                    new_results.append({
                        'Experiment_ID': folder_name,
                        'Method': method,
                        'Dataset': dataset,
                        'Hash': hash_val,
                        'Timestamp': timestamp,
                        'Loss_PR': loss_pr,
                        'Loss_RC': loss_rc,
                        'Loss_AUC': loss_auc,
                        'Loss_AP': loss_ap,
                        'Loss_F1': loss_f1,
                        'F1_PR': f1_pr,
                        'F1_RC': f1_rc,
                        'F1_AUC': f1_auc,
                        'F1_AP': f1_ap,
                        'F1_F1': f1_f1
                    })
                    
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

    # 6. 保存结果
    if new_results:
        # 定义列顺序
        columns = [
            'Experiment_ID', 'Method', 'Dataset', 'Hash', 'Timestamp', 
            'Loss_PR', 'Loss_RC', 'Loss_AUC', 'Loss_AP', 'Loss_F1',
            'F1_PR', 'F1_RC', 'F1_AUC', 'F1_AP', 'F1_F1'
        ]
        new_df = pd.DataFrame(new_results, columns=columns)
        
        if os.path.exists(output_csv):
            # 追加模式，不写 header
            new_df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            # 新建模式
            new_df.to_csv(output_csv, mode='w', header=True, index=False)
        
        print(f"\nSuccessfully added {len(new_results)} new records to {output_csv}")
    else:
        print("\nNo new finished experiments found.")

if __name__ == "__main__":
    summarize_results()
