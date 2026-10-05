"""查看新模型的 Top-100 特征重要性"""
import joblib, pymysql, numpy as np
from pymysql.cursors import DictCursor

# 加载模型
model = joblib.load("models/EURUSD_15m_xgboost__20260526_1917.pkl")
print(f"Model: n_estimators={model.n_estimators}")

# 获取特征名
conn = pymysql.connect(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4", cursorclass=DictCursor,
)
cur = conn.cursor()

cur.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_schema='forex_predictor' AND table_name='features_a' 
    AND column_name NOT IN ('timestamp','symbol','target_15m') 
    ORDER BY ordinal_position
""")
cols_a = [r["column_name"] for r in cur.fetchall()]

cur.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_schema='forex_predictor' AND table_name='features_b' 
    AND column_name NOT IN ('timestamp','symbol','target_15m') 
    ORDER BY ordinal_position
""")
cols_b = [r["column_name"] for r in cur.fetchall()]
cur.close()
conn.close()

all_cols = cols_a + cols_b
print(f"特征总数: {len(all_cols)}")

# 获取 XGBoost 内置重要性 (gain)
booster = model.get_booster()
importance = booster.get_score(importance_type="gain")

# 映射 f0, f1, ... 到列名
gain_map = {}
for k, v in importance.items():
    idx = int(k.replace("f", ""))
    if idx < len(all_cols):
        gain_map[idx] = v

# 排序取全部
sorted_feats = sorted(gain_map.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Feature':<40} {'Gain':>12} {'Idx'}")
print("-" * 70)
for rank, (idx, gain) in enumerate(sorted_feats, 1):
    name = all_cols[idx]
    print(f"{rank:<6} {name:<40} {gain:>12.1f} {idx}")

# 汇总
print(f"\n--- 统计 ---")
print(f"有贡献的特征: {len(gain_map)}/{len(all_cols)}")
print(f"Max gain: {sorted_feats[0][1]:.1f}")
print(f"Min gain: {sorted_feats[-1][1]:.1f}")
print(f"Median gain: {sorted_feats[len(sorted_feats)//2][1]:.1f}")
zeros = len(all_cols) - len(gain_map)
print(f"零贡献: {zeros} 个")
