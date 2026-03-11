## GAIA数据集

参考MSDS数据集的预处理方式，同时结合GAIA数据集的特点进行处理。

目前的metric数据的时间采样间隔是30s，log和trace数据的采样倒是没有固定的间隔，目前想法是，提取的这整体的30秒内的日志和追踪两个模态数据的统计特征来表示。即，GAIA数据集中的30s等价于MSDS数据集中的1s，采用这样的聚合方式进行处理。如果有其他更好的方式，可以提出。

可以采用多核处理的时候采用多核加速。

```
# 共10个service节点
dbservice1
dbservice2
logservice1
logservice2
mobservice1
mobservice2
redisservice1
redisservice2
webservice1
webservice2
```

### 日志数据

包含这十个节点的日志信息，从7月1日至7月31日

存在于MicroSS\business\business_split\business文件夹下

文件名为：`business_table_{service}_2021-07.csv`，例如：business_table_dbservice1_2021-07.csv，共有十个这样的csv文件（十个节点）

样例数据：

```csv
datetime,service,message
2021-07-01,dbservice1,"2021-07-01 10:54:22,639 | INFO | 0.0.0.4 | dbservice1 | permission_operate.py -> permission_operation -> 35 | c124e30fb40651dc | the list of all available services are redisservice1: http://0.0.0.1:9386, redisservice2: http://0.0.0.2:9387
"
2021-07-01,dbservice1,"2021-07-01 10:54:22,639 | INFO | 0.0.0.4 | dbservice1 | permission_operate.py -> permission_operation -> 51 | c124e30fb40651dc | now call service:redisservice1, inst:http://0.0.0.1:9386 as a downstream service
"
2021-07-01,dbservice1,"2021-07-01 10:54:22,696 | INFO | 0.0.0.4 | 172.17.0.3 | dbservice1 | c124e30fb40651dc | query = select passwards from username_table where user_name='ToeLCkHR'
"
```

需要对日志数据进行数据预处理，转化为日志模板，同时并从message中提取关键信息：
例如，提取日志的具体日期时间信息，并将其转为毫秒级时间戳。提取日志的log_level，例如INFO，ERROR等。

### 追踪数据

包含这十个节点的调用信息，从7月1日至7月31日

存在于MicroSS\trace\trace_split\trace文件夹下

文件名为：`trace_table_{service}_2021-07.csv`，例如：trace_table_dbservice1_2021-07.csv，共有十个这样的csv文件（十个节点）

样例数据：

```csv
timestamp,host_ip,service_name,trace_id,span_id,parent_id,start_time,end_time,url,status_code,message
2021-07-01 10:54:23,0.0.0.4,dbservice1,c124e30fb40651dc,58ac80ceea500f66,8b3e4a4003c5119c,2021-07-01 10:54:22.632751,2021-07-01 10:54:23.151922,http://0.0.0.4:9388/db_login_methods?uuid=a3036736-da17-11eb-9811-0242ac110003&user_id=ToeLCkHR,200,request call function 1 dbservice1.db_login_methods
2021-07-01 14:58:24,0.0.0.4,dbservice1,fc80cb49734064c,fa6ac0cfb325b70,bf8f0b5ac0800698,2021-07-01 14:58:03.840086,2021-07-01 14:58:24.147245,http://0.0.0.4:9388/db_login_methods?uuid=ae05ac76-da39-11eb-a832-0242ac110004&user_id=OVLDMHZf,500,request call function 1 dbservice1.db_login_methods
2021-07-01 14:58:27,0.0.0.4,dbservice1,23f4481bc3c5217f,b6fcab7926c0b307,9a7cf3c27471c4f9,2021-07-01 14:58:24.562948,2021-07-01 14:58:27.205834,http://0.0.0.4:9388/db_login_methods?uuid=ba71a776-da39-11eb-8545-0242ac110004&user_id=cwpAkUMO,200,request call function 1 dbservice1.db_login_methods
```

### 指标数据

存在于MicroSS\metric\metric_split\metric文件夹下：共有6640个文件，每个文件是一个不同的特征，但实际是有3320个特征，因为一个特征包含两个文件，分别为2021-07-01_2021-07-15的数据和2021-07-15_2021-07-31的数据。例如：dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv和dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-15_2021-07-31.csv其实是同一个特征，但是包含不同时间段的数据。

其中，dbservice1为服务实例名，0.0.0.4为IP，docker_cpu_core_0_norm_pct应该是特征名。不同的服务实例所具有的特征不完全一样，可能会具有一些特殊的特征。这十个服务节点的相关指标特征应该是docker容器的指标特征。

同时，还有一些特殊的特征（不包含在十个服务节点范围内的），比如redis_0.0.0.3_docker_cpu_core_17_norm_pct_2021-07-01_2021-07-15.csv，system_0.0.0.1_system_network_summary_tcp_TCPSackRecoveryFail_2021-07-15_2021-07-31.csv，zookeeper_0.0.0.1_docker_cpu_kernel_ticks_2021-07-01_2021-07-15.csv等，分别以`redis_,system_,zookeeper_`等开头的特征。这些可能是相应 IP 地址的主机指标。

根据metric的文件名，提取的IP与服务的映射关系如下：

```json
{
    "0.0.0.4": [
        "dbservice1",
        "mobservice2",
        "system"
    ],
    "0.0.0.2": [
        "dbservice2",
        "logservice2",
        "redisservice2",
        "system"
    ],
    "0.0.0.3": [
        "logservice1",
        "redis",
        "webservice2"
    ],
    "0.0.0.1": [
        "mobservice1",
        "redisservice1",
        "system",
        "webservice1",
        "zookeeper"
    ]
}
```

指标数据需要根据特征进行汇总，并保存为csv文件（这个需要保存全部数据，进行合并），由于数据采集的问题，可能会有一些问题需要处理，比如缺失，重复等等。这部分处理方式，需要认真分析考虑。指标数据的采集间隔为30s，这一点需要额外注意。

在数据预处理的时候，由于指标的特征过多，不同的特征大概有几百个，所以需要进行特征工程。

初步想法是去除方差很小的特征。可以给出更多的处理方法。

### 标签

对于 MicroSS 数据，数据标签设置为运行维护事件的标签，而非指标的标签。

需要对标签数据进行模板提取，并提取关键的信息，转化为下面的格式：

anomaly_type应该包括下面几种：

login failure，memory_anomalies，file moving program，normal memory freed label，access permission denied exception

```
index,datetime,service,instance,message,level,anomaly_type,st_time,ed_time,duration
0,2021-07-04,mobservice,mobservice1,"2021-07-04 00:37:11,553 | WARNING | 0.0.0.1 | 172.17.0.5 | mobservice1 | 48ccc1781d26a064 | wait for 11 seconds for follow-up operations to simulate the login failure of the QR code expired",WARNING,[login failure],2021-07-04 00:37:11.553000,2021-07-04 00:37:22.553000,11
1,2021-07-04,mobservice,mobservice2,"2021-07-04 00:48:32,796 | WARNING | 0.0.0.4 | 172.17.0.2 | mobservice2 | [memory_anomalies] trigger a high memory program, start at 2021-07-04 00:38:14.418677 and lasts 600 seconds and use 1g memory",WARNING,[memory_anomalies],2021-07-04 00:38:14.418677,2021-07-04 00:48:14.418677,600
3,2021-07-04,webservice,webservice2,"2021-07-04 01:10:00,821 | WARNING | 0.0.0.3 | 172.17.0.4 | webservice2 | trigger the file moving program, start with 2021-07-04 01:00:00.639653, last for 600 seconds",WARNING,[file moving program],2021-07-04 01:00:00.639653,2021-07-04 01:10:00.639653,600
54,2021-07-05,webservice,webservice2,"2021-07-05 08:25:27,738 | WARNING | 0.0.0.3 | 172.17.0.4 | webservice2 | [normal memory freed label] lasts ten minutes",WARNING,[normal memory freed label],2021-07-05 08:25:27.738000,2021-07-05 08:35:27.738000,600
342,2021-07-13,dbservice,dbservice1,"2021-07-13 10:46:12,912 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 | trigger an access permission denied exception, will lasts an hour",WARNING,[access permission denied exception],2021-07-13 10:46:12.912000,2021-07-13 11:46:12.912000,3600

```



### 关于该数据集的一些Issues

- question 1: There are duplicate timestamps in many metrics. Some of these duplicates have the same value, but often the same timestamp appears in multiple rows with different values. Usually in such cases, one of these rows has a valid value and the remaining rows are 0. Can I just take the non-zero row as the correct row to use for this timestamp? Is this expected when you collected and compiled the data? Thanks.

  - answer 1: First, in general, this situation is normal because there may be some uncertainty in the data collection process, resulting in multiple records being recorded under the same timestamp.
    Second, some metrics in GAIA dataset (mainly those starting with "system" in the filename) were recorded without tags, resulting in data from different time series being recorded together. In this case, it is necessary to perform aggregation operations on the metric data based on their specific situation. For example, for the "system_network_out_dropped" metric, it can be aggregated using the sum function.

- question 2： Some injected memory anomalies do not have impact on the memory metrics
  For example, I am checking the impact of the following injected anomaly:
  2021-08-14 03:49:04,212 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 | [memory_anomalies] trigger a high memory program, start at 2021-08-14 03:39:03.575551 and lasts 600 seconds and use 1g memory
  Below is a plot of the memory-related metrics on this node and service around that time. Red region is the duration of the high-memory program.
  There is no change in any metric before and after the anomaly is injected.
  - answer 2:Thank you for your concern to GAIA-Dataset. If the metrics of the Docker container cannot reflect the problem, you can try to view the host metrics of the corresponding IP address.