import json, os
from datetime import datetime

BASE = r'D:\dapanAllThings\codes\forex_predictor\dashboard\data\_extract'
OUT = r'D:\dapanAllThings\codes\forex_predictor\dashboard\data\page-data.json'

mm = json.load(open(os.path.join(BASE, 'model_meta.json'), encoding='utf-8'))
sig = json.load(open(os.path.join(BASE, 'latest_signals.json'), encoding='utf-8'))
shap = json.load(open(os.path.join(BASE, 'shap_top.json'), encoding='utf-8'))
dh = json.load(open(os.path.join(BASE, 'data_health.json'), encoding='utf-8'))

page = {
    "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "source_note": "FALLBACK=使用现有模型产物",
    "model_meta": {
        "name": mm["name"],
        "trained_at": mm["trained_at"],
        "auc": mm["auc"],
        "features_n": mm["features_n"],
        "currency_pairs": mm["currency_pairs"],
        "model_file": mm["model_file"],
    },
    "latest_signals": sig["latest_signals"],
    "shap_top": shap["shap_top"],
    "data_health": {
        "m1_latest_ts": dh["m1_latest_ts"],
        "coverage_days": dh["coverage_days"],
        "gap_hours": dh["gap_hours"],
        "source": dh["source"],
    },
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(page, f, ensure_ascii=False, indent=2)
print('written:', OUT)
print('top keys:', list(page.keys()))
