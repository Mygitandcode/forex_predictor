import json, os, time, urllib.request
from datetime import datetime

BASE = r'D:\dapanAllThings\codes\forex_predictor\dashboard\data\_extract'
os.makedirs(BASE, exist_ok=True)
MODELS = r'D:\dapanAllThings\codes\forex_predictor\models'
now = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---- 1. model_meta（FALLBACK：E1 现役产物）----
pkl = os.path.join(MODELS, r'EURUSD_15m_xgboost__EURUSD_15m_E1.pkl')
meta = json.load(open(os.path.join(MODELS, 'EURUSD_15m_xgboost__EURUSD_15m_E1_meta.json'), encoding='utf-8'))
trained_at = datetime.fromtimestamp(os.path.getmtime(pkl)).strftime('%Y-%m-%d %H:%M:%S')
model_meta = {
    "name": "EURUSD_15m_xgboost__EURUSD_15m_E1",
    "trained_at": trained_at,
    "auc": 0.529711,
    "features_n": meta["n_cols"],
    "currency_pairs": ["EURUSD"],
    "model_file": pkl,
}
json.dump(model_meta, open(os.path.join(BASE, 'model_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[1] model_meta OK | trained_at(=pkl mtime):', trained_at, '| features_n:', meta['n_cols'])
print('    auc=0.529711 溯源: model_runs run_id=3a84118f739f4194be9590dcf44382f2 (dimension=60, version 20260521_mtf, created 2026-08-02 14:57:48, _pipeline_status --json 输出)')

# ---- 2. latest_signals → null（两路径实证不可用）----
sig = {"latest_signals": None, "note": "realtime_predictor 为 ZeroMQ 常驻循环（--symbol 启动即 while True，无单轮模式，且无 ZMQ 数据源运行）；model_predictions 表 COUNT(*)=0（2026-09-05 23:48 经 mysql_proxy/pymysql 实查）→ 按辅助C 规则整块置 null"}
json.dump(sig, open(os.path.join(BASE, 'latest_signals.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[2] latest_signals → null（predictor=常驻循环 + predictions 表 0 行，双实证）')

# ---- 3. shap_top → null（脚本实跑失败，报错实录）----
shap = {"shap_top": None, "note": "_shap_report.py 实跑失败（2026-09-05 23:48）：shap PermutationExplainer ValueError: max_evals=500 is too low, must be at least 2*num_features+1=1695（当前 FeatureStore 特征数 ~847，脚本参数停在旧特征库时代）。按辅助C 规则整块置 null；修法建议进待澄清"}
json.dump(shap, open(os.path.join(BASE, 'shap_top.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[3] shap_top → null（实跑 ValueError 证据在案）')

# ---- 4. data_health（SQL 直查溯源）----
import pymysql
conn = pymysql.connect(host='localhost', port=3305, user='root', password='', database='forex_predictor', charset='utf8mb4')
cur = conn.cursor()
cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM candles_1m WHERE symbol='EURUSD'")
mn, mx = cur.fetchone()
cur.close(); conn.close()
mx_dt = mx if isinstance(mx, datetime) else datetime.fromisoformat(str(mx))
mn_dt = mn if isinstance(mn, datetime) else datetime.fromisoformat(str(mn))
coverage_days = round((mx_dt - mn_dt).total_seconds() / 86400, 1)
gap_hours = round((datetime.now() - mx_dt).total_seconds() / 3600, 1)
dh = {
    "m1_latest_ts": mx_dt.strftime('%Y-%m-%d %H:%M:%S'),
    "coverage_days": coverage_days,
    "gap_hours": gap_hours,
    "source": "MySQL candles_1m (EURUSD)",
}
json.dump(dh, open(os.path.join(BASE, 'data_health.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[4] data_health OK | MIN:', mn_dt, '| MAX:', mx_dt, '| coverage_days:', coverage_days, '| gap_hours:', gap_hours)

print('\nEXTRACT DONE at', now(), '→', BASE)
