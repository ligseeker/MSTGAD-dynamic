"""
GAIA数据集预处理脚本
将原始GAIA数据（MicroSS）转换为MSTGAD所需的中间格式。

输出到 data/GAIA-pre/ 目录：
  - label_events.csv  : 提取后的异常事件表（含起止时间、异常类型等）
  - label.csv          : 逐30s时间步标签（timestamp + 各节点标签）
  - log.csv            : 日志模板聚合数据
  - trace.csv          : 追踪调用数据
  - trace_path.pkl     : 服务依赖邻接矩阵
  - metric.csv         : 指标数据（归一化后，各节点维度统一）
  - preprocess.log     : 预处理日志

Usage:
    python -m util.GAIA.pre_GAIA
    python -m util.GAIA.pre_GAIA --step=2       # 只运行标签处理
    python -m util.GAIA.pre_GAIA --force         # 强制重新运行所有步骤
"""

import os
import re
import sys
import gc
import time
import pickle
import logging
import warnings
from io import StringIO
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from util.GAIA.constant import *

# ============ 路径配置 ============
Raw_Path = './data/GAIA/MicroSS'
Save_Path = './data/GAIA-pre'

BUSINESS_DIR = os.path.join(Raw_Path, 'business', 'business_split', 'business')
TRACE_DIR = os.path.join(Raw_Path, 'trace', 'trace_split', 'trace')
METRIC_DIR = os.path.join(Raw_Path, 'metric', 'metric_split', 'metric')
LABEL_DIR = os.path.join(Raw_Path, 'run', 'run', 'run')

# 时区常量：GAIA数据中的datetime字符串均为北京时间(UTC+8)
GAIA_TZ = 'Asia/Shanghai'

logger = logging.getLogger(__name__)


def _setup_logging(save_path):
    """配置logging：同时输出到控制台和文件"""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'preprocess.log')

    # 清除已有handlers（避免重复）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")


def _read_truncated_csv(filepath, **kwargs):
    """
    安全读取可能在中间截断的CSV文件。
    GAIA样例数据只保留了前几千行，可能在引号字段中间截断，
    导致pandas解析卡死。此函数先检测并剥离末尾不完整的记录。
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')
    quote_count = 0
    last_complete_line = len(lines) - 1

    for i, line in enumerate(lines):
        quote_count += line.count('"')

    if quote_count % 2 != 0:
        while last_complete_line > 0:
            quote_count -= lines[last_complete_line].count('"')
            last_complete_line -= 1
            if quote_count % 2 == 0:
                break

    complete_content = '\n'.join(lines[:last_complete_line + 1])
    return pd.read_csv(StringIO(complete_content), **kwargs)


def _ts_str_to_ms(ts_str, ms_frac=0):
    """
    将GAIA中的北京时间字符串转换为epoch毫秒。
    所有GAIA的datetime字符串都是Asia/Shanghai时区。
    """
    ts = pd.Timestamp(ts_str, tz=GAIA_TZ)
    return int(ts.timestamp() * 1000) + ms_frac


# ====================================================================
#  1. 标签处理
# ====================================================================

def parse_anomaly_event(row):
    """
    从run_table的一行解析异常事件信息。
    支持的异常类型：
      WARNING: memory_anomalies, normal memory freed label, login failure,
               file moving program, access permission denied exception,
               cpu_anomalies
      ERROR:   所有ERROR事件标记为异常（持续时间为0，仅标记所在30s窗口）
      INFO:    正常事件，不标记
    """
    msg = str(row['message']).strip()
    service = str(row['service']).strip()
    datetime_str = str(row['datetime']).strip()

    # 提取日志时间（精确到毫秒）
    ts_match = re.match(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),(\d{3})', msg)
    if ts_match:
        log_time_str = ts_match.group(1)
        log_time_ms = int(ts_match.group(2))
    else:
        log_time_str = None
        log_time_ms = 0

    # 提取log_level
    level_match = re.search(r'\|\s*(WARNING|ERROR|INFO|DEBUG)\s*\|', msg)
    level = level_match.group(1) if level_match else 'UNKNOWN'

    # 提取instance（服务名）
    instance = service

    # 识别异常类型和计算时间范围
    anomaly_type = 'unknown'
    st_time = None
    ed_time = None
    duration = 0

    if level == 'INFO':
        # INFO级别为正常事件
        anomaly_type = 'normal'
        return {
            'datetime': datetime_str, 'service': service, 'instance': instance,
            'message': msg, 'level': level, 'anomaly_type': '[normal]',
            'st_time': '', 'ed_time': '', 'duration': 0,
        }

    elif level == 'ERROR':
        # ERROR事件：标记为异常，持续时间为0（仅标记所在30s窗口）
        anomaly_type = 'error_event'
        if log_time_str:
            st_time = pd.Timestamp(log_time_str, tz=GAIA_TZ)
            ed_time = st_time  # 持续时间为0
            duration = 0

    elif level == 'WARNING':
        # WARNING级别需要解析具体异常类型

        if '[memory_anomalies]' in msg:
            anomaly_type = 'memory_anomalies'
            start_match = re.search(r'start at (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', msg)
            dur_match = re.search(r'lasts\s+(\d+)\s+seconds', msg)
            if start_match:
                st_time = pd.Timestamp(start_match.group(1), tz=GAIA_TZ)
            if dur_match:
                duration = int(dur_match.group(1))
            if st_time and duration:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        elif '[cpu_anomalies]' in msg:
            anomaly_type = 'cpu_anomalies'
            # 格式与memory_anomalies类似
            start_match = re.search(r'start at (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', msg)
            dur_match = re.search(r'lasts\s+(\d+)\s+seconds', msg)
            if start_match:
                st_time = pd.Timestamp(start_match.group(1), tz=GAIA_TZ)
            if dur_match:
                duration = int(dur_match.group(1))
            if st_time and duration:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        elif '[normal memory freed label]' in msg:
            anomaly_type = 'normal memory freed label'
            if log_time_str:
                st_time = pd.Timestamp(f"{log_time_str}.{log_time_ms:03d}", tz=GAIA_TZ)
            duration = 600  # ten minutes
            if st_time:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        elif 'login failure' in msg or 'simulate the login failure' in msg:
            anomaly_type = 'login failure'
            dur_match = re.search(r'wait for (\d+) seconds', msg)
            if log_time_str:
                st_time = pd.Timestamp(f"{log_time_str}.{log_time_ms:03d}", tz=GAIA_TZ)
            if dur_match:
                duration = int(dur_match.group(1))
            else:
                duration = 11
            if st_time:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        elif 'file moving program' in msg or 'trigger the file moving program' in msg:
            anomaly_type = 'file moving program'
            start_match = re.search(r'start with (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', msg)
            dur_match = re.search(r'last for (\d+) seconds', msg)
            if start_match:
                st_time = pd.Timestamp(start_match.group(1), tz=GAIA_TZ)
            if dur_match:
                duration = int(dur_match.group(1))
            if st_time and duration:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        elif 'access permission denied exception' in msg:
            anomaly_type = 'access permission denied exception'
            if log_time_str:
                st_time = pd.Timestamp(f"{log_time_str}.{log_time_ms:03d}", tz=GAIA_TZ)
            if 'an hour' in msg:
                duration = 3600
            else:
                dur_match = re.search(r'lasts\s+(\d+)\s+seconds', msg)
                duration = int(dur_match.group(1)) if dur_match else 3600
            if st_time:
                ed_time = st_time + pd.Timedelta(seconds=duration)

        else:
            # 未识别的WARNING类型，打印用于调试
            anomaly_type = 'unknown_warning'
            if log_time_str:
                st_time = pd.Timestamp(f"{log_time_str}.{log_time_ms:03d}", tz=GAIA_TZ)
                ed_time = st_time
                duration = 0

    return {
        'datetime': datetime_str,
        'service': service,
        'instance': instance,
        'message': msg,
        'level': level,
        'anomaly_type': f'[{anomaly_type}]',
        'st_time': str(st_time) if st_time else '',
        'ed_time': str(ed_time) if ed_time else '',
        'duration': duration,
    }


def deal_label(label_path, save_path, metric_timestamps=None):
    """
    处理标签数据：
    1. 解析run_table，提取异常事件 -> label_events.csv
    2. 生成逐30s时间步的标签矩阵 -> label.csv
       - 时间戳与metric对齐（对齐到30s网格）
       - INFO为正常，WARNING/ERROR中有异常跨度的30s窗口标记为1
    3. 生成逐30s时间步的异常类型辅助标签
       - label_type.csv: 每个服务节点在该窗口内对应的异常类型字符串
    """
    logger.info("Processing labels...")
    label_file = os.path.join(label_path, 'run_table_2021-07.csv')
    df = _read_truncated_csv(label_file)

    # 解析所有异常事件
    events = []
    unknown_types = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Parsing label events'):
        event = parse_anomaly_event(row)
        event['index'] = idx
        events.append(event)
        if event['anomaly_type'] == '[unknown]' or event['anomaly_type'] == '[unknown_warning]':
            # 截取关键部分用于调试
            short_msg = event['message'][:150]
            unknown_types.add(f"{event['level']}|{short_msg}")

    if unknown_types:
        logger.warning(f"Found {len(unknown_types)} unknown event types:")
        for ut in list(unknown_types)[:10]:
            logger.warning(f"  {ut}")

    events_df = pd.DataFrame(events)
    events_df = events_df[['index', 'datetime', 'service', 'instance', 'message',
                            'level', 'anomaly_type', 'st_time', 'ed_time', 'duration']]
    events_df.to_csv(os.path.join(save_path, 'label_events.csv'), index=False, encoding='utf-8')
    logger.info(f"Saved label_events.csv with {len(events_df)} events")

    # 统计异常类型分布
    type_counts = events_df['anomaly_type'].value_counts()
    logger.info(f"Anomaly type distribution:\n{type_counts.to_string()}")

    # 构建30秒对齐的时间轴
    interval_ms = GAIA_SAMPLE_INTERVAL * 1000  # 30000ms

    if metric_timestamps is not None:
        mt = np.array(metric_timestamps)
        first_ts = mt[0]
        base_ts = (first_ts // interval_ms) * interval_ms
        aligned = ((mt - base_ts) // interval_ms) * interval_ms + base_ts
        timestamps = sorted(set(aligned.tolist()))
        logger.info(f"Metric timestamps aligned to 30s grid: {len(metric_timestamps)} -> {len(timestamps)} unique steps")
        logger.info(f"  First: {timestamps[0]} ({pd.Timestamp(timestamps[0]/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)})")
        logger.info(f"  Last:  {timestamps[-1]} ({pd.Timestamp(timestamps[-1]/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)})")
    else:
        start_ts = int(pd.Timestamp('2021-07-01 00:00:00', tz=GAIA_TZ).timestamp() * 1000)
        end_ts = int(pd.Timestamp('2021-07-31 23:59:59', tz=GAIA_TZ).timestamp() * 1000)
        timestamps = list(range(start_ts, end_ts + 1, interval_ms))

    # 初始化标签矩阵
    label_matrix = pd.DataFrame(0, index=timestamps, columns=GAIA_SERVICES)
    label_matrix.index.name = 'timestamp'
    label_type_matrix = pd.DataFrame('[normal]', index=timestamps, columns=GAIA_SERVICES)
    label_type_matrix.index.name = 'timestamp'

    # 将异常事件映射到时间步
    # 筛选非normal事件（st_time和ed_time都有值的）
    anomaly_events = events_df[
        (events_df['st_time'] != '') &
        (events_df['ed_time'] != '') &
        (events_df['anomaly_type'] != '[normal]')
    ].copy()
    logger.info(f"Mapping {len(anomaly_events)} anomaly events to 30s windows...")

    anomaly_types = sorted(anomaly_events['anomaly_type'].unique().tolist())

    for _, event in anomaly_events.iterrows():
        service = event['instance']
        if service not in GAIA_SERVICES:
            continue

        st_str = event['st_time']
        ed_str = event['ed_time']

        # 将时间字符串转为epoch ms（st_time已带时区信息）
        st = pd.Timestamp(st_str)
        ed = pd.Timestamp(ed_str)
        # 跳过NaT（解析失败的情况）
        if pd.isna(st) or pd.isna(ed):
            continue
        st_ms = int(st.timestamp() * 1000)
        ed_ms = int(ed.timestamp() * 1000)

        # 对齐到30s网格
        st_ms_aligned = (st_ms // interval_ms) * interval_ms
        ed_ms_aligned = (ed_ms // interval_ms) * interval_ms

        # 标记异常时间段：该30s窗口内有异常即标为1
        if st_ms_aligned == ed_ms_aligned:
            # ERROR等瞬时事件，只标记一个窗口
            if st_ms_aligned in label_matrix.index:
                label_matrix.loc[st_ms_aligned, service] = 1
                current_type = label_type_matrix.loc[st_ms_aligned, service]
                if current_type == '[normal]':
                    label_type_matrix.loc[st_ms_aligned, service] = event['anomaly_type']
                elif event['anomaly_type'] not in current_type.split('|'):
                    label_type_matrix.loc[st_ms_aligned, service] = current_type + '|' + event['anomaly_type']
        else:
            mask = (label_matrix.index >= st_ms_aligned) & (label_matrix.index <= ed_ms_aligned)
            label_matrix.loc[mask, service] = 1
            matched_timestamps = label_matrix.index[mask]
            for ts in matched_timestamps:
                current_type = label_type_matrix.loc[ts, service]
                if current_type == '[normal]':
                    label_type_matrix.loc[ts, service] = event['anomaly_type']
                elif event['anomaly_type'] not in current_type.split('|'):
                    label_type_matrix.loc[ts, service] = current_type + '|' + event['anomaly_type']

    label_matrix.to_csv(os.path.join(save_path, 'label.csv'), index=True)
    logger.info(f"Saved label.csv with {len(label_matrix)} timesteps")

    label_type_matrix.to_csv(os.path.join(save_path, 'label_type.csv'), index=True)
    logger.info(f"Saved label_type.csv with {len(label_type_matrix)} timesteps")

    # 统计
    for svc in GAIA_SERVICES:
        n_anomaly = (label_matrix[svc] == 1).sum()
        logger.info(f"  {svc}: {n_anomaly} anomalous / {len(label_matrix)} total ({100*n_anomaly/len(label_matrix):.2f}%)")

    type_cover_counts = {}
    for anomaly_type in anomaly_types:
        type_cover_counts[anomaly_type] = int(
            label_type_matrix.apply(lambda col: col.astype(str).str.contains(anomaly_type, regex=False)).values.sum()
        )
    if type_cover_counts:
        logger.info("Aligned anomaly type coverage:")
        for anomaly_type, count in sorted(type_cover_counts.items(), key=lambda item: (-item[1], item[0])):
            logger.info(f"  {anomaly_type}: {count}")

    return label_matrix


# ====================================================================
#  2. 日志处理
# ====================================================================

def deal_log(business_path, save_path):
    """
    处理日志数据：
    1. 读取10个business CSV文件
    2. 从message提取时间戳（注意时区：北京时间）和log_level
    3. 用Drain3进行日志模板挖掘
    4. 按30s窗口聚合模板计数、log level计数和总日志量
    """
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    import jsonpickle

    logger.info("Processing logs...")

    config_file = os.path.join(os.path.dirname(__file__), 'gaia.ini')
    config = TemplateMinerConfig()
    config.load(config_file)
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    all_logs = []
    parse_fail_count = 0
    for svc in GAIA_SERVICES:
        filename = f'business_table_{svc}_2021-07.csv'
        filepath = os.path.join(business_path, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Missing log file: {filepath}")
            continue
        df = _read_truncated_csv(filepath, keep_default_na=False)
        logger.info(f"  Read {len(df)} log entries for {svc}")

        for _, row in df.iterrows():
            msg = str(row['message']).strip()
            service = str(row['service']).strip()

            # 提取精确时间戳（北京时间）
            ts_match = re.match(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),(\d{3})', msg)
            if ts_match:
                ts_str = ts_match.group(1)
                ts_ms_frac = int(ts_match.group(2))
                try:
                    # 关键修复：指定时区为Asia/Shanghai
                    timestamp_ms = _ts_str_to_ms(ts_str, ts_ms_frac)
                except Exception:
                    parse_fail_count += 1
                    continue
            else:
                parse_fail_count += 1
                continue

            # 提取log_level
            level_match = re.search(r'\|\s*(WARNING|ERROR|INFO|DEBUG)\s*\|', msg)
            log_level = level_match.group(1) if level_match else 'UNKNOWN'

            # 提取消息体（最后一个 | 后面的内容）用于模板挖掘
            parts = msg.split('|')
            if len(parts) >= 2:
                payload = parts[-1].strip()
            else:
                payload = msg

            # Drain3 模板挖掘
            result = template_miner.add_log_message(payload)
            template_id = result['cluster_id']

            all_logs.append({
                'timestamp_ms': timestamp_ms,
                'service': service,
                'log_level': log_level,
                'template_id': template_id,
                'payload': payload,
            })

    if parse_fail_count > 0:
        logger.warning(f"Failed to parse timestamp for {parse_fail_count} log entries")

    log_df = pd.DataFrame(all_logs)
    num_templates = len(template_miner.drain.clusters)
    logger.info(f"Total log entries: {len(log_df)}, Templates: {num_templates}")

    # 按30s窗口聚合：对齐到30s整数倍
    interval_ms = GAIA_SAMPLE_INTERVAL * 1000
    log_df['timestamp_aligned'] = (log_df['timestamp_ms'] // interval_ms) * interval_ms

    # 统计日志时间范围
    if len(log_df) > 0:
        ts_min = log_df['timestamp_aligned'].min()
        ts_max = log_df['timestamp_aligned'].max()
        logger.info(f"Log time range: {pd.Timestamp(ts_min/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)} ~ "
                     f"{pd.Timestamp(ts_max/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)}")
        n_windows = log_df['timestamp_aligned'].nunique()
        logger.info(f"Log covers {n_windows} unique 30s windows")

    # 构建聚合结果
    log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'UNKNOWN']
    logger.info("Aggregating templates and levels...")
    group_keys = ['timestamp_aligned', 'service']

    template_counts = (
        log_df.groupby(group_keys + ['template_id'])
        .size()
        .unstack(fill_value=0)
    )
    template_counts.columns = [f'template_{c}' for c in template_counts.columns]

    level_counts = (
        log_df.groupby(group_keys + ['log_level'])
        .size()
        .unstack(fill_value=0)
    )
    level_counts.columns = [f'level_{c}' for c in level_counts.columns]

    # Ensure all expected level columns exist for downstream loading.
    expected_level_cols = [f'level_{level}' for level in log_levels]
    level_counts = level_counts.reindex(columns=expected_level_cols, fill_value=0)

    total_counts = log_df.groupby(group_keys).size().to_frame('log_total')

    agg_df = pd.concat([template_counts, level_counts, total_counts], axis=1).reset_index()
    agg_df = agg_df.rename(columns={'timestamp_aligned': 'timestamp'})
    agg_df['timestamp'] = agg_df['timestamp'].astype(np.int64)

    count_cols = [c for c in agg_df.columns if c not in ['timestamp', 'service']]
    agg_df[count_cols] = agg_df[count_cols].fillna(0).astype(np.int64)
    agg_df = agg_df.sort_values(['timestamp', 'service']).reset_index(drop=True)

    # 保存
    agg_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)
    logger.info(f"Saved log.csv with {len(agg_df)} aggregated records")
    logger.info(
        "Log feature groups: "
        f"templates={num_templates}, levels={len(log_levels)}, extras=1(total)"
    )

    # 保存模板信息（使用 tab 分隔避免模板中含逗号的问题）
    template_file = os.path.join(save_path, 'log_templates.tsv')
    with open(template_file, 'w') as f:
        f.write('template_id\ttemplate\n')
        for cluster in template_miner.drain.clusters:
            f.write(f'{cluster.cluster_id}\t{cluster.get_template()}\n')
    logger.info(f"Saved {num_templates} templates to log_templates.tsv")

    # 保存Drain3状态以便复用
    state_file = os.path.join(save_path, 'log_state.pkl')
    state = jsonpickle.dumps(template_miner.drain, keys=True).encode('utf-8')
    with open(state_file, 'wb') as f:
        f.write(state)

    return num_templates


# ====================================================================
#  3. 追踪处理
# ====================================================================

def deal_trace(trace_path, save_path):
    """
    处理追踪数据：
    1. 读取10个trace CSV文件（注意时区修复）
    2. 构建服务调用关系
    3. 生成调用特征
    """
    logger.info("Processing traces...")

    all_traces = []
    for svc in GAIA_SERVICES:
        filename = f'trace_table_{svc}_2021-07.csv'
        filepath = os.path.join(trace_path, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Missing trace file: {filepath}")
            continue
        df = _read_truncated_csv(filepath, keep_default_na=False)
        logger.info(f"  Read {len(df)} trace entries for {svc}")
        all_traces.append(df)

    trace_df = pd.concat(all_traces, ignore_index=True)

    # 关键修复：计算 duration（秒），指定时区为Asia/Shanghai
    trace_df['start_time_ts'] = pd.to_datetime(trace_df['start_time']).dt.tz_localize(GAIA_TZ)
    trace_df['end_time_ts'] = pd.to_datetime(trace_df['end_time']).dt.tz_localize(GAIA_TZ)
    trace_df['duration'] = (trace_df['end_time_ts'] - trace_df['start_time_ts']).dt.total_seconds()

    # 确保 status_code 是字符串
    trace_df['status_code'] = trace_df['status_code'].astype(str)

    trace_types = sorted(trace_df['status_code'].unique().tolist())
    logger.info(f"Trace types (status codes): {trace_types}")

    # 对齐timestamp到30s窗口（使用修正后的时区时间戳）
    interval_ms = GAIA_SAMPLE_INTERVAL * 1000
    trace_df['timestamp_aligned'] = trace_df['end_time_ts'].apply(
        lambda x: int(x.timestamp() * 1000) // interval_ms * interval_ms
    )

    # 统计追踪时间范围
    ts_min = trace_df['timestamp_aligned'].min()
    ts_max = trace_df['timestamp_aligned'].max()
    logger.info(f"Trace time range: {pd.Timestamp(ts_min/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)} ~ "
                 f"{pd.Timestamp(ts_max/1000, unit='s', tz='UTC').tz_convert(GAIA_TZ)}")
    n_windows = trace_df['timestamp_aligned'].nunique()
    n_days = (ts_max - ts_min) / (86400 * 1000)
    logger.info(f"Trace covers {n_windows} unique 30s windows over {n_days:.1f} days")
    logger.info(f"Average records per window: {len(trace_df)/n_windows:.1f}")

    # 构建服务调用关系矩阵，并保留真实的跨服务边记录
    logger.info("Building service call relationships...")

    call_pairs = set()
    edge_records = []
    for trace_id, group in tqdm(trace_df.groupby('trace_id'), desc='Building call graph'):
        span_map = {}
        for _, span in group.iterrows():
            span_map[span['span_id']] = span['service_name']

        for _, span in group.iterrows():
            parent = span['parent_id']
            child_svc = span['service_name']
            if parent in span_map:
                parent_svc = span_map[parent]
                if parent_svc != child_svc and parent_svc in GAIA_SERVICES and child_svc in GAIA_SERVICES:
                    call_pairs.add((parent_svc, child_svc))
                    edge_records.append({
                        'timestamp_aligned': int(span['timestamp_aligned']),
                        'src_service': parent_svc,
                        'dst_service': child_svc,
                        'status_code': str(span['status_code']),
                        'duration': float(span['duration']),
                    })

    # 构建邻接矩阵
    relation_matrix = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)))
    for (src, dst) in call_pairs:
        i, j = GAIA_SERVICES.index(src), GAIA_SERVICES.index(dst)
        relation_matrix[i, j] = 1
        relation_matrix[j, i] = 1  # 无向图
    logger.info(f"Service call pairs ({len(call_pairs)}): {call_pairs}")
    logger.info(f"Adjacency matrix edges: {int(relation_matrix.sum())}")

    # 保存邻接矩阵
    pickle.dump(relation_matrix, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))

    # 构建 trace 边特征聚合，保留 src->dst 方向信息
    if edge_records:
        trace_edge_df = pd.DataFrame(edge_records)
        trace_agg = trace_edge_df.groupby(
            ['timestamp_aligned', 'src_service', 'dst_service', 'status_code']
        ).agg(
            duration_sum=('duration', 'sum'),
            duration_mean=('duration', 'mean'),
            count=('duration', 'count'),
        ).reset_index()
    else:
        trace_agg = pd.DataFrame(
            columns=[
                'timestamp_aligned', 'src_service', 'dst_service', 'status_code',
                'duration_sum', 'duration_mean', 'count'
            ]
        )

    trace_agg.to_csv(os.path.join(save_path, 'trace.csv'), index=False)
    logger.info(f"Saved trace.csv with {len(trace_agg)} aggregated directed edge records")

    return trace_types, relation_matrix


# ====================================================================
#  4. 指标处理
# ====================================================================

def _parse_metric_filename(filename):
    """
    解析metric文件名，提取service, ip, feature_name, 时间范围
    例: dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv
    """
    name = filename.replace('.csv', '')
    time_match = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$', name)
    if not time_match:
        return None
    date_start = time_match.group(1)
    date_end = time_match.group(2)
    prefix = name[:time_match.start()]

    svc_ip_match = re.match(r'^(\w+)_([\d.]+)_(.+)$', prefix)
    if not svc_ip_match:
        return None

    service = svc_ip_match.group(1)
    ip = svc_ip_match.group(2)
    feature = svc_ip_match.group(3)

    return {
        'service': service,
        'ip': ip,
        'feature': feature,
        'date_start': date_start,
        'date_end': date_end,
        'full_name': f'{service}_{ip}_{feature}',
    }


def _target_services_for_metric(info):
    """
    将原始metric映射到服务节点。
    - 容器指标: 直接映射到对应服务节点
    - system指标: 根据IP挂到同机服务节点，并显式加 host_ 前缀
    """
    service = info['service']
    if service in GAIA_SERVICES:
        return [(service, info['feature'])]

    if service == 'system':
        mapped_services = [svc for svc in GAIA_IP_MAP.get(info['ip'], []) if svc in GAIA_SERVICES]
        return [(svc, f'host_{info["feature"]}') for svc in mapped_services]

    return []


def _metric_duplicate_reduce_mode(feature_name):
    """根据metric语义选择重复时间戳的聚合方式。"""
    feature = feature_name.lower()

    sum_keywords = [
        'network', 'bytes', 'packets', 'dropped', 'errors', 'fault',
        'ops', 'operations', 'request', 'requests', 'sectors',
        'switches', 'total_', '_total', 'count', 'connections',
    ]
    max_keywords = [
        'max', 'peak', 'highest', 'watermark',
    ]

    if any(keyword in feature for keyword in max_keywords):
        return 'max'
    if any(keyword in feature for keyword in sum_keywords):
        return 'sum'
    return 'mean'


def _reduce_duplicate_timestamp_values(values, reduce_mode):
    """
    聚合同一timestamp下的重复记录。
    优先避免“一个非零值被多个零值稀释”的情况。
    """
    arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    non_zero = arr[np.abs(arr) > 1e-12]
    if non_zero.size == 1 and arr.size > 1:
        return float(non_zero[0])

    target = non_zero if non_zero.size > 0 else arr
    if reduce_mode == 'sum':
        return float(target.sum())
    if reduce_mode == 'max':
        return float(target.max())
    return float(target.mean())


def _metric_quality_stats(series, total_timestamps, interval_ms):
    """计算单个metric特征的覆盖度、动态性和稳定性统计量。"""
    values = pd.to_numeric(series.values, errors='coerce')
    values = values[np.isfinite(values)]

    if total_timestamps <= 0 or values.size == 0:
        return {
            'coverage': 0.0,
            'unique_count': 0,
            'non_zero_ratio': 0.0,
            'dynamic_span': 0.0,
            'dynamic_ratio': 0.0,
            'score': 0.0,
        }

    aligned_index = ((series.index.values.astype(np.int64) // interval_ms) * interval_ms)
    coverage = float(len(np.unique(aligned_index)) / total_timestamps)
    unique_count = int(pd.Series(values).nunique(dropna=True))
    non_zero_ratio = float((np.abs(values) > 1e-12).mean())

    if values.size == 1:
        q05 = q95 = float(values[0])
    else:
        q05, q95 = np.quantile(values, [0.05, 0.95])
    dynamic_span = float(q95 - q05)
    mean_abs = float(np.mean(np.abs(values)))
    dynamic_ratio = float(dynamic_span / (mean_abs + 1e-6))

    score = coverage * (0.5 + 0.5 * non_zero_ratio) * (
        np.log1p(dynamic_span) + 0.1 * np.log1p(unique_count)
    )

    return {
        'coverage': coverage,
        'unique_count': unique_count,
        'non_zero_ratio': non_zero_ratio,
        'dynamic_span': dynamic_span,
        'dynamic_ratio': dynamic_ratio,
        'score': float(score),
    }


def _merge_metric_pair(args):
    """合并同一特征的两个时段文件"""
    feature_name, file_pair, feature_alias = args
    dfs = []
    for filepath in file_pair:
        try:
            df = pd.read_csv(filepath)
            if 'timestamp' in df.columns and 'value' in df.columns:
                dfs.append(df)
        except Exception as e:
            pass

    if not dfs:
        return feature_name, None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['timestamp', 'value'])
    merged['value'] = pd.to_numeric(merged['value'], errors='coerce')
    merged = merged.dropna(subset=['timestamp', 'value'])

    if merged.duplicated(subset=['timestamp']).any():
        reduce_mode = _metric_duplicate_reduce_mode(feature_alias)
        merged = merged.groupby('timestamp', as_index=False)['value'].agg(
            lambda values: _reduce_duplicate_timestamp_values(values, reduce_mode)
        )

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    return feature_name, merged


def deal_metric(metric_path, save_path):
    """
    处理指标数据：
    1. 读取10个服务节点的docker指标，并将system host指标挂接到对应服务
    2. 合并两个时段文件，并按metric类型处理重复时间戳
    3. 多核CPU特征汇总
    4. 先做质量筛选，再统一各节点维度（稳定共享特征交集）
    5. 方差过滤
    6. Min-Max归一化
    """
    logger.info("Processing metrics...")

    # 扫描所有文件
    files = os.listdir(metric_path)
    feature_files = defaultdict(list)
    parsed_info = {}
    raw_service_count = 0
    host_service_links = 0

    for f in files:
        info = _parse_metric_filename(f)
        if info is None:
            continue
        targets = _target_services_for_metric(info)
        if not targets:
            continue
        full_name = info['full_name']
        feature_files[full_name].append(os.path.join(metric_path, f))
        parsed_info[full_name] = info
        if info['service'] == 'system':
            host_service_links += len(targets)
        else:
            raw_service_count += 1

    logger.info(
        f"Found {len(feature_files)} unique raw metric features "
        f"({raw_service_count} service metrics, {host_service_links} host-to-service links)"
    )

    # 并行合并时段文件
    merge_args = [(name, files, parsed_info[name]['feature']) for name, files in feature_files.items()]
    n_workers = min(cpu_count(), 8)
    logger.info(f"Merging metric files with {n_workers} workers...")
    merged_features = {}
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(_merge_metric_pair, merge_args),
                           total=len(merge_args), desc='Merging metrics'))
    for name, df in results:
        if df is not None and len(df) > 0:
            merged_features[name] = df

    logger.info(f"Successfully merged {len(merged_features)} features")

    # 收集所有有效的时间戳
    logger.info("Collecting timestamps...")
    ts_arrays = [df['timestamp'].values for df in merged_features.values()]
    all_timestamps = np.unique(np.concatenate(ts_arrays))
    logger.info(f"Total unique timestamps: {len(all_timestamps)}")

    # 多核CPU特征汇总 + system指标挂接到服务节点
    logger.info("Aggregating multi-core features...")
    # service_features: {service: {base_feature_name: [full_names]}}
    # 其中 base_feature_name 去掉了 service_ip_ 前缀的部分用于跨服务对齐
    service_features = defaultdict(lambda: defaultdict(list))
    # 同时维护 base_feature -> (col_name, full_names) 映射以便跨服务对齐
    service_col_map = defaultdict(dict)  # {service: {base_feature: col_name}}

    for full_name, info in parsed_info.items():
        if full_name not in merged_features:
            continue
        ip = info['ip']
        for service, feature in _target_services_for_metric(info):
            is_multicore = False
            for prefix in GAIA_MULTI_CORE_PREFIXES:
                if prefix in feature:
                    base = re.sub(r'core_\d+', 'core_X', feature)
                    col_name = f'{service}_{ip}_{base}'
                    service_features[service][col_name].append(full_name)
                    service_col_map[service][base] = col_name
                    is_multicore = True
                    break

            if not is_multicore:
                col_name = f'{service}_{ip}_{feature}'
                service_features[service][col_name].append(full_name)
                service_col_map[service][feature] = col_name

    # 构建每个特征的 Series
    logger.info("Building feature Series...")
    feature_series_dict = {}

    for service in GAIA_SERVICES:
        features_dict = service_features[service]
        for base_name, full_names in features_dict.items():
            if len(full_names) == 1:
                df = merged_features[full_names[0]]
                s = pd.Series(df['value'].values, index=df['timestamp'].values, name=base_name)
            else:
                core_series = []
                for fn in full_names:
                    if fn in merged_features:
                        df = merged_features[fn]
                        core_series.append(
                            pd.Series(df['value'].values, index=df['timestamp'].values)
                        )
                if not core_series:
                    continue
                combined = pd.concat(core_series, axis=1)
                s = combined.mean(axis=1)
                s.name = base_name
            feature_series_dict[base_name] = s

    logger.info(f"After multi-core aggregation: {len(feature_series_dict)} features total")

    # ========== 统一各节点维度 ==========
    # 先做质量筛选，再对稳定共享特征取交集，避免直接做并集导致高稀疏和无信息特征混入
    logger.info("Filtering low-quality metric features before alignment...")
    interval_ms = GAIA_SAMPLE_INTERVAL * 1000
    aligned_timestamps = np.unique((all_timestamps // interval_ms) * interval_ms)

    min_feature_coverage = 0.20
    min_dynamic_ratio = 1e-4
    min_unique_values = 2
    feature_quality = defaultdict(dict)
    svc_base_features = {}
    filtered_reason_counter = defaultdict(int)

    for svc in GAIA_SERVICES:
        kept_features = set()
        for base_feat, col_name in service_col_map[svc].items():
            if col_name not in feature_series_dict:
                continue
            stats = _metric_quality_stats(feature_series_dict[col_name], len(aligned_timestamps), interval_ms)
            feature_quality[svc][base_feat] = stats

            if stats['coverage'] < min_feature_coverage:
                filtered_reason_counter['low_coverage'] += 1
                continue
            if stats['unique_count'] < min_unique_values:
                filtered_reason_counter['constant'] += 1
                continue
            if stats['dynamic_span'] < 1e-10 or stats['dynamic_ratio'] < min_dynamic_ratio:
                filtered_reason_counter['low_dynamic'] += 1
                continue

            kept_features.add(base_feat)

        svc_base_features[svc] = kept_features
        logger.info(f"  {svc}: kept {len(kept_features)} / {len(service_col_map[svc])} features after quality filtering")

    if filtered_reason_counter:
        logger.info(
            "Filtered feature reasons: " +
            ", ".join(f"{reason}={count}" for reason, count in sorted(filtered_reason_counter.items()))
        )

    if any(len(svc_base_features[svc]) == 0 for svc in GAIA_SERVICES):
        logger.warning("Quality filtering removed all features for at least one service; falling back to unfiltered shared intersection")
        svc_base_features = {
            svc: set(service_col_map[svc].keys())
            for svc in GAIA_SERVICES
        }

    common_features = set.intersection(*svc_base_features.values())
    logger.info(f"Stable shared features across all services (intersection): {len(common_features)}")

    # 按交集筛选，构建统一的特征列（每个服务使用相同顺序的特征）
    common_features_sorted = sorted(common_features)
    unified_series = {}
    for svc in GAIA_SERVICES:
        for base_feat in common_features_sorted:
            col_name = service_col_map[svc][base_feat]
            # 重命名列：{service}_{base_feat}（统一命名规则）
            unified_col = f'{svc}_{base_feat}'
            if col_name in feature_series_dict:
                s = feature_series_dict[col_name].copy()
                s.name = unified_col
                unified_series[unified_col] = s

    logger.info(f"Unified features: {len(unified_series)} ({len(common_features_sorted)} per node x {len(GAIA_SERVICES)} nodes)")

    # 使用pd.concat一次性合并
    logger.info("Concatenating all features into DataFrame...")
    metric_result = pd.concat(unified_series.values(), axis=1)
    metric_result.index.name = 'timestamp'
    metric_result = metric_result.sort_index()

    # 释放内存
    del merged_features, feature_series_dict, unified_series, ts_arrays
    gc.collect()

    # 短间隔缺失前向填充
    max_fill = 10
    logger.info(f"Forward filling (limit={max_fill})...")
    metric_result = metric_result.ffill(limit=max_fill)

    # 方差过滤（按base_feature统一过滤，保证各节点维度一致）
    # 只有当某个base_feature在【所有】节点上都是低方差时才移除
    variance_threshold = 1e-10
    variances = metric_result.var()

    # 提取每列的base_feature名（去掉service前缀）
    base_feat_low_var_count = defaultdict(int)  # base_feature -> 低方差的节点数
    base_feat_total_count = defaultdict(int)    # base_feature -> 总节点数
    col_to_base = {}
    for col in metric_result.columns:
        for svc in GAIA_SERVICES:
            if col.startswith(svc + '_'):
                base = col[len(svc) + 1:]
                col_to_base[col] = base
                base_feat_total_count[base] += 1
                if variances[col] < variance_threshold:
                    base_feat_low_var_count[base] += 1
                break

    # 只移除在所有节点上都低方差的base_feature
    remove_base_feats = set()
    for base, count in base_feat_low_var_count.items():
        if count == base_feat_total_count[base]:
            remove_base_feats.add(base)

    if remove_base_feats:
        remove_cols = [col for col, base in col_to_base.items() if base in remove_base_feats]
        logger.info(f"Removing {len(remove_base_feats)} base features ({len(remove_cols)} columns) "
                     f"that are low-variance across all nodes (var < {variance_threshold})")
        metric_result = metric_result.drop(columns=remove_cols)

    # Min-Max 归一化
    logger.info("Min-Max normalizing...")
    col_min = metric_result.min()
    col_max = metric_result.max()
    col_range = col_max - col_min
    col_range = col_range.replace(0, 1)
    metric_result = (metric_result - col_min) / col_range

    # 统计每个节点的最终特征数
    feature_cols = metric_result.columns.tolist()
    for svc in GAIA_SERVICES:
        svc_cols = [c for c in feature_cols if c.startswith(svc + '_')]
        logger.info(f"  {svc}: {len(svc_cols)} features")

    # 将index变为列再保存
    metric_result = metric_result.reset_index()
    metric_result.to_csv(os.path.join(save_path, 'metric.csv'), index=False)
    logger.info(f"Saved metric.csv: shape={metric_result.shape}")

    cache_file = os.path.join(save_path, 'metric_aggregated.pkl')
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info(f"Removed stale metric cache: {cache_file}")

    return metric_result


# ====================================================================
#  5. 主函数
# ====================================================================

def main():
    os.makedirs(Save_Path, exist_ok=True)
    _setup_logging(Save_Path)

    # 解析命令行参数
    run_steps = set()
    force = False
    for arg in sys.argv[1:]:
        if arg.startswith('--step='):
            steps = arg.split('=')[1].split(',')
            run_steps = set(int(s) for s in steps)
        elif arg == '--force':
            force = True
    run_all = len(run_steps) == 0

    def should_run(step_num, output_file):
        """判断是否需要运行某步骤"""
        if force:
            return run_all or step_num in run_steps
        return (run_all or step_num in run_steps) and not os.path.exists(output_file)

    # 1. 处理指标数据
    metric_csv = os.path.join(Save_Path, 'metric.csv')
    if should_run(1, metric_csv):
        logger.info("=" * 60)
        logger.info("Step 1: Processing metric data")
        logger.info("=" * 60)
        metric_result = deal_metric(METRIC_DIR, Save_Path)
    elif os.path.exists(metric_csv):
        logger.info("Step 1: metric.csv already exists, loading...")
        metric_result = pd.read_csv(metric_csv)
    else:
        logger.info("Step 1: skipped")
        metric_result = None

    # 2. 处理标签数据
    label_csv = os.path.join(Save_Path, 'label.csv')
    if should_run(2, label_csv):
        logger.info("=" * 60)
        logger.info("Step 2: Processing label data")
        logger.info("=" * 60)
        if metric_result is not None:
            metric_timestamps = metric_result['timestamp'].tolist()
        else:
            metric_timestamps = None
        deal_label(LABEL_DIR, Save_Path, metric_timestamps=metric_timestamps)
    else:
        logger.info("Step 2: label.csv already exists or skipped")

    # 3. 处理日志数据
    log_csv_path = os.path.join(Save_Path, 'log.csv')
    if should_run(3, log_csv_path):
        logger.info("=" * 60)
        logger.info("Step 3: Processing log data")
        logger.info("=" * 60)
        num_templates = deal_log(BUSINESS_DIR, Save_Path)
    elif os.path.exists(log_csv_path):
        templates_file = os.path.join(Save_Path, 'log_templates.tsv')
        if os.path.exists(templates_file):
            with open(templates_file) as f:
                num_templates = sum(1 for _ in f) - 1
        else:
            # 兼容旧版csv格式
            templates_file_csv = os.path.join(Save_Path, 'log_templates.csv')
            if os.path.exists(templates_file_csv):
                with open(templates_file_csv) as f:
                    num_templates = sum(1 for _ in f) - 1
            else:
                num_templates = '?'
    else:
        logger.info("Step 3: skipped")
        num_templates = '?'

    # 4. 处理追踪数据
    trace_csv_path = os.path.join(Save_Path, 'trace.csv')
    if should_run(4, trace_csv_path):
        logger.info("=" * 60)
        logger.info("Step 4: Processing trace data")
        logger.info("=" * 60)
        trace_types, relation_matrix = deal_trace(TRACE_DIR, Save_Path)
    elif os.path.exists(trace_csv_path):
        logger.info("Step 4: trace.csv already exists")
        trace_types = '?'
        relation_matrix = None
    else:
        logger.info("Step 4: skipped")
        trace_types = '?'
        relation_matrix = None

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Output directory: {Save_Path}")
    if metric_result is not None:
        logger.info(f"  Metric features: {metric_result.shape[1] - 1}")
    logger.info(f"  Log templates: {num_templates}")
    logger.info(f"  Trace types: {trace_types}")
    if relation_matrix is not None:
        logger.info(f"  Graph edges: {int(relation_matrix.sum())}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
