"""
特征选择分析：用最佳参数训练模型 → SHAP 全局重要性 → 决策保留/丢弃
用法: python _feature_select.py
"""
import sys
import os
import gc
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).parent))

from features.feature_store import FeatureStore

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STUDY_NAME = "EURUSD_15m_xgboost__v2_80t"
STORAGE = f"sqlite:///optuna_{STUDY_NAME}.db"
DATA_DIR = "E:/forex_npy"
SYMBOL = "EURUSD"
TARGET = "15m"
N_FEATURES = 1060

# ---- 1. 加载数据 ----
logger.info("Loading data (mmap)...")
store = FeatureStore(DATA_DIR)
data = store.load_all(SYMBOL, mmap_mode='r')
X_full = data['features']
y_full = data.get(TARGET)
if y_full is None:
    y_full = data.get(f'labels_{TARGET}')
if y_full is None:
    y_full = data.get('labels')
if y_full is None:
    raise KeyError(f"Label '{TARGET}' not found. Available: {list(data.keys())}")

# clamp invalid
y_full = np.clip(y_full, 0, 1).astype(int)
logger.info(f"Data: X={X_full.shape}, pos_rate={y_full.mean():.1%}")

# ---- 2. 时序切分（和训练一致） ----
split_idx = int(len(X_full) * 0.8)
X_train, X_test = X_full[:split_idx], X_full[split_idx:]
y_train, y_test = y_full[:split_idx], y_full[split_idx:]
logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---- 3. 加载最佳参数 ----
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
best = study.best_trial
logger.info(f"Best Trial #{best.number}: AUC={best.value:.4f}, params={best.params}")

# ---- 4. 用最佳参数训练全量模型 ----
from xgboost import XGBClassifier

params = {k: v for k, v in best.params.items()
          if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                    'colsample_bytree', 'min_child_weight', 'reg_alpha',
                    'reg_lambda', 'gamma']}
params.update({'n_jobs': 1, 'random_state': 42, 'verbosity': 1})

logger.info(f"Training model with best params...")
params['early_stopping_rounds'] = 100
model = XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
n_trees = model.get_booster().num_boosted_rounds()
logger.info(f"Model trained. trees={n_trees} (early stopping)")

# ---- 5. SHAP 全局特征重要性 ----
logger.info("Computing SHAP (TreeExplainer)...")
import shap

t_shap = time.time()
# Sample for SHAP to keep it fast
shap_sample_size = min(5000, len(X_test))
shap_sample = X_test[:shap_sample_size]

try:
    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(shap_sample)
    importance = np.abs(shap_values).mean(axis=0)
    method = "TreeExplainer"
except Exception as e:
    logger.warning(f"TreeExplainer failed ({e}), falling back to PermutationExplainer")
    explainer = shap.Explainer(
        lambda x: model.predict_proba(x)[:, 1],
        shap_sample[:min(500, shap_sample_size)],
        max_evals=3000,
    )
    shap_result = explainer(shap_sample[:min(200, shap_sample_size)])
    importance = np.abs(shap_result.values).mean(axis=0)
    method = "PermutationExplainer"

elapsed = time.time() - t_shap
logger.info(f"SHAP done ({method}, {elapsed:.0f}s)")

# ---- 6. 排序 + 累积贡献分析 ----
sorted_idx = np.argsort(importance)[::-1]
sorted_imp = importance[sorted_idx]
cumsum = np.cumsum(sorted_imp)
total = cumsum[-1]
cum_pct = cumsum / total * 100

# 找各个阈值的特征数
thresholds = [50, 70, 80, 90, 95, 99]
threshold_counts = {}
for thresh in thresholds:
    n = int(np.searchsorted(cum_pct, thresh)) + 1
    threshold_counts[thresh] = n

# 输出文件
report_path = Path(DATA_DIR) / SYMBOL / "shap_feature_analysis_report.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

lines = []  # 收集所有输出行
def out(s=""):
    print(s)
    lines.append(s)

out("=" * 70)
out(f"SHAP Feature Importance Analysis ({N_FEATURES} features)")
out("=" * 70)
out(f"\n  Method: {method}")
out(f"  Total absolute SHAP: {total:.6f}")
out(f"  Max single feature:  {sorted_imp[0]:.6f} (f{sorted_idx[0]})")
out(f"  Feature 100:         {sorted_imp[99]:.6f}")
out(f"  Feature 200:         {sorted_imp[199]:.6f}")
out(f"  Feature 500:         {sorted_imp[499]:.6f}")
out(f"  Feature 1000:        {sorted_imp[999]:.6f} (last 60 are effectively zero)")

out(f"\n  Cumulative contribution thresholds:")
for thresh, n in threshold_counts.items():
    out(f"    Top {n:4d} features -> {thresh:2d}% of total importance")
    if thresh == 90:
        last_imp = sorted_imp[n-1] if n <= len(sorted_imp) else 0
        out(f"           (last included importance: {last_imp:.8f})")

# ---- 7. 详细输出 Top-200 (完整信息) ----
out(f"\n{'='*70}")
out(f"Top-200 Features (ranked by SHAP importance)")
out(f"{'='*70}")
out(f"{'Rank':<6} {'Feature':<10} {'SHAP':<14} {'Cum%':<10} {'Segment'}")
out("-" * 70)

for rank, (idx, imp) in enumerate(zip(sorted_idx[:200], sorted_imp[:200]), 1):
    cum = cum_pct[rank - 1]
    if rank <= 100:
        seg = "KEEP"
    elif rank <= 200:
        seg = "BORDERLINE"
    else:
        seg = "DROP"
    out(f"{rank:<6} f{idx:<9} {imp:<14.8f} {cum:<10.2f}% {seg}")

# ---- 8. 区段分析 ----
out(f"\n{'='*70}")
out(f"Segment Analysis")
out(f"{'='*70}")
out(f"  Top-100 contribution: {cum_pct[99]:.2f}%")
out(f"  101-200 contribution: {cum_pct[199] - cum_pct[99]:.2f}%")
out(f"  201-500 contribution: {cum_pct[499] - cum_pct[199]:.2f}%")
out(f"  501-1060 contribution: {100 - cum_pct[499]:.2f}%")

# 每 50 个特征一段的分析
out(f"\n  Per-50-feature breakdown:")
for start in range(0, 500, 50):
    end = min(start + 50, N_FEATURES)
    seg_contrib = cum_pct[end-1] - (cum_pct[start-1] if start > 0 else 0)
    out(f"    f{start}:{end:<4d} -> +{seg_contrib:.2f}% (cumulative: {cum_pct[end-1]:.2f}%)")

# ---- 9. 决策建议 ----
out(f"\n{'='*70}")
out(f"RECOMMENDATION")
out(f"{'='*70}")
n_90 = threshold_counts[90]
n_95 = threshold_counts[95]
out(f"""
  Situation: {N_FEATURES}-dim AUC ceiling = {best.value:.4f}
  
  Tier 1 (Keep):      Top-100 features -> {cum_pct[99]:.1f}% contribution
                       Expected: equal or slightly lower AUC
                       Benefit: 10x fewer features = faster training
  
  Tier 2 (Try):       Top-50  features -> {cum_pct[49]:.1f}% contribution
                       If AUC stays stable, 50 features is enough
  
  Tier 3 (Maybe):     Top-200 features -> {cum_pct[199]:.1f}% contribution
                       Diminishing returns: +{cum_pct[199]-cum_pct[99]:.1f}% for 2x features
  
  Next: The last {N_FEATURES - 500} features contribute only {100-cum_pct[499]:.1f}%
  -> Pure noise. Definite cut.
""")

# 写出报告文件
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
logger.info(f"Full report written to {report_path}")

# ---- 10. 保存 JSON 完整排名 ----
output = {
    "best_auc": float(best.value),
    "best_params": best.params,
    "n_features": N_FEATURES,
    "shap_method": method,
    "thresholds": {str(k): v for k, v in threshold_counts.items()},
    "top200_shap": [
        {"rank": i+1, "feature_index": int(sorted_idx[i]), "importance": float(sorted_imp[i])}
        for i in range(200)
    ],
    "all_features_ranked": [int(x) for x in sorted_idx.tolist()],
}
output_path = report_path.parent / "shap_feature_importance.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info(f"JSON data saved to {output_path}")

# ---- 11. 生成 Top-100 索引列表供后续训练 ----
selected_features = sorted(int(x) for x in sorted_idx[:100])
features_path = report_path.parent / "selected_top100_features.json"
with open(features_path, "w", encoding="utf-8") as f:
    json.dump({"selected_indices": selected_features, "k": 100, "source_study": STUDY_NAME}, f)
logger.info(f"Selected feature indices saved to {features_path}")

# ---- 12. 额外：Top-50 和 Top-90% 也保存 ----
for k, name in [(50, "top50"), (n_90, f"top{int(90)}pct"), (n_95, f"top{int(95)}pct")]:
    sel = sorted(int(x) for x in sorted_idx[:k])
    p = report_path.parent / f"selected_{name}_features.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"selected_indices": sel, "k": k, "source_study": STUDY_NAME}, f)

del model
gc.collect()
logger.info(f"Done. All reports in {report_path.parent}")
logger.info(f"  - {report_path.name} (full text report)")
logger.info(f"  - shap_feature_importance.json (full ranking data)")
logger.info(f"  - selected_top100_features.json (indices for retraining)")
