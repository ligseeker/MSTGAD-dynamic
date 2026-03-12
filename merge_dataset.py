import os
import re
import pickle
import argparse
from tqdm import tqdm

def merge_dataset(target_dir):
    """
    将由于历史原因生成的由散落小 .pkl 文件组成的数据集目录，
    合并为一个大的 dataset.pkl 文件以进行高速的大促IO加载。
    
    Args:
        target_dir: 要合并的目录（例如 './MSDS-save' 或 './GAIA-save'）
    """
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.")
        return

    dataset_file = os.path.join(target_dir, 'dataset.pkl')
    if os.path.exists(dataset_file):
        print(f"Watch out: {dataset_file} already exists. Please delete it if you want to perform a fresh merge.")
        return

    print(f"Scanning directory {target_dir}...")
    dataset = os.listdir(target_dir)
    # 仅过滤 .pkl 文件，忽略可能存在的其余文件或目录
    dataset = [f for f in dataset if f.endswith('.pkl')]
    dataset.sort(key=lambda x: (int(re.split(r"[-_.]", x)[0])))
    
    if not dataset:
        print(f"No small .pkl files found in {target_dir}.")
        return

    print(f"Found {len(dataset)} small .pkl files. Starting merge process...")
    merged_data = []
    
    for file in tqdm(dataset):
        file_path = os.path.join(target_dir, file)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # 剔除无关紧要但占体积且无需的 'name'
                if 'name' in data:
                    del data['name']
                merged_data.append(data)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    print(f"Saving exactly {len(merged_data)} samples to {dataset_file}...")
    try:
        with open(dataset_file, 'wb') as f:
            pickle.dump(merged_data, f)
        print("Merge successfully complete! You can now safely delete the individual small .pkl files.")
        print(f"Path to unified file: {dataset_file}")
    except Exception as e:
        print(f"Error saving {dataset_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge many small .pkl files into a single dataset.pkl for fast disk IO.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the small .pkl files (e.g., ./MSDS-save)")
    args = parser.parse_args()
    merge_dataset(args.dir)
