# GAIA数据集 - 10个服务节点
GAIA_SERVICES = [
    'dbservice1', 'dbservice2',
    'logservice1', 'logservice2',
    'mobservice1', 'mobservice2',
    'redisservice1', 'redisservice2',
    'webservice1', 'webservice2',
]

# IP 到服务的映射关系（根据metric文件名提取）
GAIA_IP_MAP = {
    '0.0.0.4': ['dbservice1', 'mobservice2', 'system'],
    '0.0.0.2': ['dbservice2', 'logservice2', 'redisservice2', 'system'],
    '0.0.0.3': ['logservice1', 'redis', 'webservice2'],
    '0.0.0.1': ['mobservice1', 'redisservice1', 'system', 'webservice1', 'zookeeper'],
}

# 异常类型列表
GAIA_ANOMALY_TYPES = [
    'login failure',
    'memory_anomalies',
    'cpu_anomalies',
    'file moving program',
    'normal memory freed label',
    'access permission denied exception',
]

# 时间采样间隔（秒）
GAIA_SAMPLE_INTERVAL = 30

# 需要汇总的多核CPU特征前缀（将多核合并为单一均值特征）
GAIA_MULTI_CORE_PREFIXES = [
    'docker_cpu_core_',
]
