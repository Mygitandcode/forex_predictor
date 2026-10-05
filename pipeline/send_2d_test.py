"""发送 2 维 MQ 测试消息 (快速验证多维度耗时)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.nlu import parse
from pipeline.schema import validate_request, to_json
from pipeline.mq import RedisStreamClient, check_redis
from pipeline.feature_selector import get_top_features

features = get_top_features(2)
print(f"特征: {features}")

config = {
    "trailName": "二维测试_2d",
    "symbol": "EURUSD",
    "target_tf": "15m",
    "features": features,
    "n_trials": 1,
    "holdout": False,
    "mode": "train",
    "created_by": "2维耗时测试",
}
ok, err, norm = validate_request(config)
if not ok:
    print(f"FAIL: {err}")
    sys.exit(1)
print(f"配置: {norm['trailName']} features={len(norm['features'])} trials={norm['n_trials']}")

if not check_redis():
    print("ERROR: Redis 不可用")
    sys.exit(1)

mq = RedisStreamClient()
msg_id = mq.push_request(norm)
print(f"已发送: {norm['trailName']}")
