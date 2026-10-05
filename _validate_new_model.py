"""验证新模型 (1060维) 在 2026-01~04 上的表现"""
import pymysql, joblib, numpy as np
from pymysql.cursors import DictCursor
from sklearn.metrics import roc_auc_score, accuracy_score

# 加载新模型
model = joblib.load("models/EURUSD_15m_xgboost__20260526_1917.pkl")
print(f"Model: n_estimators={model.n_estimators}")

# 连接 MySQL
conn = pymysql.connect(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4", cursorclass=DictCursor,
)
cur = conn.cursor()

# 获取列名
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
print(f"特征列: a={len(cols_a)}, b={len(cols_b)}, total={len(cols_a)+len(cols_b)}")

# 查询 2026-01~04 数据 (列名以数字开头，需反引号)
a_cols = ",".join(["a.`" + c + "`" for c in cols_a])
b_cols = ",".join(["b.`" + c + "`" for c in cols_b])
sql = f"""
    SELECT a.timestamp, a.target_15m, {a_cols}, {b_cols}
    FROM features_a a JOIN features_b b ON a.timestamp=b.timestamp
    WHERE a.timestamp >= '2026-01-01' AND a.timestamp <= '2026-04-30'
    ORDER BY a.timestamp
"""
cur.execute(sql)

X, y = [], []
for r in cur.fetchall():
    vals = [float(r[c] if r[c] is not None else 0) for c in cols_a]
    vals += [float(r[c] if r[c] is not None else 0) for c in cols_b]
    X.append(vals)
    y.append(int(r["target_15m"]))
cur.close()
conn.close()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int8)
valid = y >= 0
X, y = X[valid], y[valid]
print(f"2026-01~04 有效样本: {len(y):,}")

# 预测
y_prob = model.predict_proba(X)[:, 1]
y_sig = np.full(len(y), -1, dtype=int)
y_sig[y_prob >= 0.7] = 1
y_sig[y_prob <= 0.3] = 0
has_sig = y_sig >= 0
n_sig = has_sig.sum()
sig_acc = (y_sig[has_sig] == y[has_sig]).sum() / max(n_sig, 1) if n_sig > 0 else 0

# 分方向
up_mask = y_sig == 1
dn_mask = y_sig == 0
up_acc = (y_sig[up_mask] == y[up_mask]).sum() / max(up_mask.sum(), 1) if up_mask.sum() > 0 else 0
dn_acc = (y_sig[dn_mask] == y[dn_mask]).sum() / max(dn_mask.sum(), 1) if dn_mask.sum() > 0 else 0

print(f"\n======== 新模型 (1060维, AUC 0.5283) — 2026-01~04 ========")
print(f"  AUC:        {roc_auc_score(y, y_prob):.4f}")
print(f"  Accuracy:   {accuracy_score(y, (y_prob>=0.5).astype(int)):.4f}")
print(f"  信号数:      {n_sig:,} ({n_sig/len(y)*100:.1f}%)")
print(f"  信号准确率:   {sig_acc:.4f}")
print(f"  涨信号:      {up_mask.sum()} 准确={up_acc:.4f}")
print(f"  跌信号:      {dn_mask.sum()} 准确={dn_acc:.4f}")
print(f"==========================================================")
