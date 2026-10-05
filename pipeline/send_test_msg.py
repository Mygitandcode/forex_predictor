"""
发送一条测试 MQ 消息到 training:requests, 供 orchestrator 消费。
在 PyCharm 终端运行: python pipeline/send_test_msg.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.nlu import parse
from pipeline.schema import validate_request, to_json
from pipeline.mq import RedisStreamClient, check_redis

# 自然语言 → JSON
text = "Top-1特征 训练1轮 代号可视化测试"
raw = parse(text)
ok, err, config = validate_request(raw)

if not ok:
    print(f"配置验证失败: {err}")
    sys.exit(1)

print("=" * 50)
print("MQ 消息内容:")
print(f"  trailName: {config['trailName']}")
print(f"  features:  {config['features']}")
print(f"  n_trials:  {config['n_trials']}")
print(f"  holdout:   {config['holdout']}")
print(f"  mode:      {config['mode']}")
print("=" * 50)

# 推送到 Redis Streams
if not check_redis():
    print("ERROR: Redis 不可用")
    sys.exit(1)

mq = RedisStreamClient()
msg_id = mq.push_request(config)
print(f"\n已推送到 training:requests, msg_id={msg_id}")
print("Orchestrator 将自动消费并执行全链路...")
