"""
_validate_0326.py — EURUSD 15m 模型对 2026-01~04 验证集预测评估

全自动流程:
    1. 复制原始 1~4 月数据到实验目录
    2. 加载 2025年12月历史 + 2026-01~04 数据 → 合并
    3. FeatureBuilder 向量化 (212维)
    4. 加载 XGBoost .pkl 模型预测 (joblib)
    5. 计算真实标签 (15分钟未来方向)
    6. 双阈值评估 + 输出明细
    7. 写入 mlflow_records.xlsx (主表)
    8. 输出预测详情 CSV + 评估报告 MD

输出目录: E:\forex_data\codeBuddyOutput\{date}_EURUSD_15m\
主记录:    E:\forex_data\codeBuddyOutput\mlflow_records.xlsx

使用: python _validate_0326.py
"""
import os, sys, json, shutil, gc, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from features.feature_builder import FeatureBuilder
from training.model_eval import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
SYMBOL = "EURUSD"
TARGET_WINDOW = "15m"
TARGET_MINUTES = 15
POINT_SIZE = 0.00001
THRESHOLD_UP = 0.7
THRESHOLD_DOWN = 0.3

# XGBoost .pkl 模型路径 (Stage2 产出)
MODEL_PKL = Path(r"d:\dapanAllThings\codes\forex_predictor\models\EURUSD_15m.pkl")
# 兜底: AutoGluon 模型目录
MODEL_AUTOGLUON = Path(r"d:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m")

CLEANED_CSV = Path(r"E:\forex_data\cleaneded\EURUSD\EURUSD_M1.csv")
RAW_JAN   = Path(r"E:\forex_data\active\EURUSD\2026_01\EURUSD_M1_2026_01.csv")
RAW_FEB   = Path(r"E:\forex_data\active\EURUSD\2026_02\EURUSD_M1_2026_02.csv")
RAW_MARCH = Path(r"E:\forex_data\active\EURUSD\2026_03\EURUSD_M1_2026_03.csv")
RAW_APRIL = Path(r"E:\forex_data\active\EURUSD\2026_04\EURUSD_M1_2026_04.csv")

OUTPUT_ROOT = Path(r"E:\forex_data\codeBuddyOutput")
MASTER_XLSX = OUTPUT_ROOT / "mlflow_records.xlsx"

# 实验目录: 日期_货币对_目标窗口
EXPERIMENT_DIR = OUTPUT_ROOT / f"{datetime.now().strftime('%Y%m%d')}_{SYMBOL}_{TARGET_WINDOW}"

# 历史缓冲区: 特征构建需要 200 根 + 均线需要 500 根
HISTORY_BARS = 1000


# ============================================================
# 1. 数据加载
# ============================================================
def load_feb_history(n_bars: int = HISTORY_BARS) -> pd.DataFrame:
    """从清洗 CSV 加载 2 月底最后 n_bars 根 K 线作为历史"""
    logger.info(f"Loading last {n_bars} bars from cleaned CSV...")
    # 只读最后一段，避免加载全部 6M 行
    df = pd.read_csv(CLEANED_CSV)
    df = df.tail(n_bars).copy()
    
    # 构建 datetime 索引 (格式: date=2026.02.27, time=16:58)
    datetimes = []
    for _, row in df.iterrows():
        try:
            t_str = str(row['time']).strip()
            if len(t_str) <= 5:
                t_str += ':00'
            dt = pd.Timestamp(f"{row['date']} {t_str}")
            datetimes.append(dt)
        except Exception:
            datetimes.append(pd.NaT)
    df['datetime'] = datetimes
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    df = df.rename(columns={'tick_volume': 'tick_volume'})
    # 确保列名匹配
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df = df.dropna()
    logger.info(f"  Feb history: {len(df)} bars, {df.index[0]} ~ {df.index[-1]}")
    return df


def load_raw_csv(filepath: Path, label: str) -> pd.DataFrame:
    """加载无表头的 MT4 导出 CSV"""
    logger.info(f"Loading {label}: {filepath}")
    df = pd.read_csv(
        filepath,
        header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume'],
        encoding='utf-8-sig',
    )
    logger.info(f"  {label} raw: {len(df)} rows, cols: {df.columns.tolist()}")
    
    # 构建 datetime 索引
    datetimes = []
    for _, row in df.iterrows():
        try:
            t_str = str(row['time']).strip()
            if len(t_str) <= 5:
                t_str += ':00'
            dt = pd.Timestamp(f"{row['date']} {t_str}")
            datetimes.append(dt)
        except Exception:
            datetimes.append(pd.NaT)
    
    df['datetime'] = datetimes
    df = df.dropna(subset=['datetime']).set_index('datetime')
    
    # 数值化
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce').fillna(0).astype(int)
    
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # 清洗异常值
    df = df[df['high'] >= df['low']]
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    
    logger.info(f"  {label} cleaned: {len(df)} bars, {df.index[0]} ~ {df.index[-1]}")
    return df


def merge_data(*dfs) -> pd.DataFrame:
    """合并多个DataFrame，去重排序"""
    merged = pd.concat(dfs)
    merged = merged[~merged.index.duplicated(keep='first')]
    merged = merged.sort_index()
    logger.info(f"Merged: {len(merged)} bars, {merged.index[0]} ~ {merged.index[-1]}")
    return merged


# ============================================================
# 2. 特征构建 + 预测
# ============================================================
def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建特征矩阵，返回 (X, timestamps)"""
    logger.info("Building features...")
    builder = FeatureBuilder(symbol=SYMBOL, point_size=POINT_SIZE)
    X, feature_names = builder.build_many(df)
    
    # 时间戳对齐: features[0] 对应 df.index[200]
    timestamps = df.index[200:200 + len(X)].values
    
    logger.info(f"  Features: {X.shape}, dims: {len(feature_names)}")
    logger.info(f"  Timestamps: {len(timestamps)}, {timestamps[0]} ~ {timestamps[-1]}")
    
    return X.astype(np.float32), timestamps


def predict_and_label(
    X: np.ndarray,
    timestamps: np.ndarray,
    df_full: pd.DataFrame,
    original_indices: np.ndarray = None,
) -> Dict:
    """
    预测 + 计算标签 + 评估

    Args:
        X: (N, D) 特征矩阵
        timestamps: (N,) 时间戳
        df_full: 包含全部 K 线数据的 DataFrame (含历史)
        original_indices: X 中每行对应完整数据中的特征序号 (0-based)
                          df_full 中的行号 = 200 + original_indices[i]

    Returns:
        包含 predictions, labels, metrics 等结果的字典
    """
    import joblib

    # ---- 加载模型 (优先 XGBoost .pkl) ----
    if MODEL_PKL.exists():
        logger.info(f"Loading XGBoost model from {MODEL_PKL}...")
        model = joblib.load(MODEL_PKL)
        logger.info(f"  Model type: XGBoost, n_estimators={getattr(model, 'n_estimators', '?')}")

        # XGBoost 直接 predict_proba
        y_prob = model.predict_proba(X)[:, 1]
    elif MODEL_AUTOGLUON.exists():
        logger.info(f"XGBoost .pkl not found, fallback to AutoGluon: {MODEL_AUTOGLUON}")
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(str(MODEL_AUTOGLUON))
        logger.info(f"  Model type: {predictor.problem_type}, eval_metric: {predictor.eval_metric}")

        feature_names = [f"f{i}" for i in range(X.shape[1])]
        test_df = pd.DataFrame(X, columns=feature_names)
        y_pred_proba = predictor.predict_proba(test_df)
        y_prob = y_pred_proba.iloc[:, 1].values if y_pred_proba.shape[1] > 1 else y_pred_proba.values.flatten()
    else:
        raise FileNotFoundError(
            f"No model found. Checked:\n"
            f"  XGBoost: {MODEL_PKL} (exists={MODEL_PKL.exists()})\n"
            f"  AutoGluon: {MODEL_AUTOGLUON} (exists={MODEL_AUTOGLUON.exists()})"
        )

    logger.info("Predicting...")

    # 双阈值信号
    y_signal = np.full_like(y_prob, -1, dtype=int)
    y_signal[y_prob >= THRESHOLD_UP] = 1    # 涨
    y_signal[y_prob <= THRESHOLD_DOWN] = 0  # 跌

    # 计算真实标签: 当前 close vs 15分钟后 close
    closes = df_full['close'].values
    n = len(X)
    y_true = np.full(n, -1, dtype=np.int8)
    future_closes = np.full(n, np.nan, dtype=np.float64)
    actual_changes = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        source_offset = original_indices[i] if original_indices is not None else i
        source_idx = 200 + source_offset
        future_idx = source_idx + TARGET_MINUTES

        if future_idx < len(closes):
            y_true[i] = 1 if closes[future_idx] > closes[source_idx] else 0
            future_closes[i] = closes[future_idx]
            actual_changes[i] = (closes[future_idx] - closes[source_idx]) / closes[source_idx]

    logger.info(f"  Labels: {len(y_true)} total, {int((y_true >= 0).sum())} valid (not -1)")

    # 评估
    evaluator = ModelEvaluator(threshold_up=THRESHOLD_UP, threshold_down=THRESHOLD_DOWN)
    metrics = evaluator.evaluate(y_true, y_prob, timestamps, symbol=SYMBOL)

    return {
        'timestamps': timestamps,
        'y_prob': y_prob,
        'y_signal': y_signal,
        'y_true': y_true,
        'future_closes': future_closes,
        'actual_changes': actual_changes,
        'metrics': metrics,
    }


# ============================================================
# 3. 输出结果
# ============================================================
def save_detailed_csv(results: Dict, df_full: pd.DataFrame, output_dir: Path, original_indices: np.ndarray = None):
    """保存预测详情 CSV，含原数据 + 预测 + 真实 + 对比 + 汇总行"""
    n = len(results['timestamps'])
    
    # 获取对应的原始 K 线数据
    # X[i] 对应 df_full 中的行: 200 + original_indices[i]
    if original_indices is not None:
        source_indices = 200 + original_indices
    else:
        source_indices = np.arange(200, 200 + n)
    df_source = df_full.iloc[source_indices].copy()
    
    # 构建输出 DataFrame
    out = pd.DataFrame({
        'timestamp': results['timestamps'],
        'open': df_source['open'].values if 'open' in df_source.columns else np.nan,
        'high': df_source['high'].values if 'high' in df_source.columns else np.nan,
        'low': df_source['low'].values if 'low' in df_source.columns else np.nan,
        'close': df_source['close'].values if 'close' in df_source.columns else np.nan,
        'tick_volume': df_source['tick_volume'].values if 'tick_volume' in df_source.columns else 0,
        'predicted_prob': results['y_prob'],
        'predicted_signal': results['y_signal'],
        'actual_close_after_15m': results['future_closes'],
        'actual_change_pct': results['actual_changes'],
        'actual_direction': results['y_true'],
    })
    
    # 信号映射 + 判断是否正确
    def signal_label(s):
        if s == 1: return '涨'
        elif s == 0: return '跌'
        else: return '震荡'
    
    def direction_label(d):
        if d == 1: return '涨'
        elif d == 0: return '跌'
        else: return '无标签'
    
    # 正确性判断
    is_correct = []
    for s, t in zip(results['y_signal'], results['y_true']):
        if s == -1 or t == -1:
            is_correct.append('N/A')
        elif s == t:
            is_correct.append('YES')
        else:
            is_correct.append('NO')
    
    out['predicted_signal_label'] = [signal_label(s) for s in results['y_signal']]
    out['actual_direction_label'] = [direction_label(d) for d in results['y_true']]
    out['is_correct'] = is_correct
    
    # 过滤掉无标签的行做统计
    has_label = results['y_true'] >= 0
    has_signal = results['y_signal'] >= 0
    both_valid = has_label & has_signal
    
    total_valid = has_label.sum()
    total_signals = has_signal.sum()
    total_correct = int(np.sum((results['y_signal'][both_valid] == results['y_true'][both_valid])))
    accuracy = total_correct / total_signals if total_signals > 0 else 0
    
    # 汇总行
    summary_row = {
        'timestamp': '== SUMMARY ==',
        'open': '',
        'high': '',
        'low': '',
        'close': '',
        'tick_volume': '',
        'predicted_prob': '',
        'predicted_signal': f'总信号: {total_signals}',
        'actual_close_after_15m': '',
        'actual_change_pct': f'总有效标签: {total_valid}',
        'actual_direction': f'正确数: {total_correct}',
        'predicted_signal_label': f'准确率: {accuracy:.4f} ({accuracy*100:.2f}%)',
        'actual_direction_label': '',
        'is_correct': '',
    }
    out = pd.concat([out, pd.DataFrame([summary_row])], ignore_index=True)
    
    # 保存
    csv_path = output_dir / "prediction_details.csv"
    out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Detailed predictions saved: {csv_path} ({len(out)-1} rows + summary)")
    
    return out.head(len(out) - 1)  # return without summary


def save_evaluation_report(metrics: Dict, output_dir: Path):
    """保存评估报告 Markdown"""
    report = f"""# EURUSD 15m 模型验证报告

**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**货币对**: {SYMBOL}
**预测窗口**: {TARGET_WINDOW} (未来 15 分钟方向)
**验证月份**: 2026-01 ~ 2026-04
**模型路径**: {MODEL_PKL if MODEL_PKL.exists() else MODEL_AUTOGLUON}

---

## 整体指标

| 指标 | 值 |
|------|-----|
| 总样本数 | {metrics.get('n_total', 0):,} |
| 有效信号数 | {metrics.get('n_signal', 0):,} |
| 信号覆盖率 | {metrics.get('signal_ratio', 0):.2%} |
| **方向准确率** | **{metrics.get('accuracy', 0):.4f}** ({metrics.get('accuracy', 0)*100:.2f}%) |
| AUC | {metrics.get('auc', 0):.4f} |

## 分方向准确率

| 方向 | 信号数 | 准确率 |
|------|--------|--------|
| 涨 (Up) | {metrics.get('n_up_signals', 0):,} | {metrics.get('up_accuracy', 0):.4f} |
| 跌 (Down) | {metrics.get('n_down_signals', 0):,} | {metrics.get('down_accuracy', 0):.4f} |

## 滚动窗口评估

"""
    if 'rolling' in metrics and len(metrics['rolling']) > 0:
        roll = metrics['rolling']
        report += f"""
| 窗口 | 样本数 | 信号数 | 准确率 | 信号覆盖率 |
|------|--------|--------|--------|------------|
"""
        for _, row in roll.iterrows():
            report += f"| {row['date']} | {int(row['n_samples']):,} | {int(row['n_signals']):,} | {row['accuracy']:.4f} | {row['signal_ratio']:.2%} |\n"
        
        report += f"""
- **平均准确率**: {roll['accuracy'].mean():.4f}
- **准确率标准差**: {roll['accuracy'].std():.4f}
- **最高准确率**: {roll['accuracy'].max():.4f}
- **最低准确率**: {roll['accuracy'].min():.4f}
"""
    
    report += """
---

> 生成于 `_validate_0326.py`
"""
    
    md_path = output_dir / "evaluation_report.md"
    md_path.write_text(report, encoding='utf-8')
    logger.info(f"Evaluation report saved: {md_path}")


def update_master_xlsx(metrics: Dict, experiment_dir: Path):
    """更新 mlflow_records.xlsx 主记录表"""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    
    MASTER_XLSX.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'experiment_id', 'symbol', 'target_window', 'predict_months',
        'model_path', 'created_at', 'n_samples', 'n_signals',
        'signal_ratio', 'accuracy', 'auc', 'up_accuracy', 'down_accuracy',
        'n_up_signals', 'n_down_signals', 'experiment_folder'
    ]
    
    # 本次实验 ID = 时间戳
    exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    row_data = [
        exp_id,
        SYMBOL,
        TARGET_WINDOW,
        '2026-01, 2026-02, 2026-03, 2026-04',
        str(MODEL_PKL if MODEL_PKL.exists() else MODEL_AUTOGLUON),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metrics.get('n_total', 0),
        metrics.get('n_signal', 0),
        round(metrics.get('signal_ratio', 0), 4),
        round(metrics.get('accuracy', 0), 4),
        round(metrics.get('auc', 0), 4),
        round(metrics.get('up_accuracy', 0), 4),
        round(metrics.get('down_accuracy', 0), 4),
        metrics.get('n_up_signals', 0),
        metrics.get('n_down_signals', 0),
        str(experiment_dir),
    ]
    
    # 打开或创建
    if MASTER_XLSX.exists():
        wb = openpyxl.load_workbook(MASTER_XLSX)
        ws = wb['experiments'] if 'experiments' in wb.sheetnames else wb.active
        if ws.title != 'experiments':
            ws.title = 'experiments'
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'experiments'
        # 写入表头
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
    
    # 追加数据
    next_row = ws.max_row + 1
    for col_idx, val in enumerate(row_data, 1):
        cell = ws.cell(row=next_row, column=col_idx, value=val)
        cell.alignment = Alignment(horizontal='center')
    
    # 自动列宽
    for col_idx in range(1, len(headers) + 1):
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ''))
            for r in range(1, next_row + 1)
        )
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = min(max_len + 4, 30)
    
    wb.save(MASTER_XLSX)
    logger.info(f"Master record updated: {MASTER_XLSX} row {next_row} (exp_id={exp_id})")
    
    return exp_id


# ============================================================
# 主流程
# ============================================================
def main():
    print("\n" + "=" * 60)
    print(f"  EURUSD {TARGET_WINDOW} 验证 — 2026-03/04 预测评估")
    print("=" * 60)
    
    # ---- 0. 创建目录 ----
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment dir: {EXPERIMENT_DIR}")
    
    # ---- 1. 复制原始数据到实验目录 ----
    logger.info("Copying raw data files...")
    for f in [RAW_JAN, RAW_FEB, RAW_MARCH, RAW_APRIL]:
        shutil.copy2(f, EXPERIMENT_DIR / f.name)
    
    # ---- 2. 加载数据 ----
    # History from late 2025 (last 2000 bars of training data, from cleaned CSV)
    dec_df = load_feb_history(HISTORY_BARS * 2)  # double buffer, filter to Dec 2025
    dec_df = dec_df[dec_df.index >= pd.Timestamp('2025-12-01')]
    logger.info(f"Dec 2025 history: {len(dec_df)} bars")
    
    jan_df = load_raw_csv(RAW_JAN, "2026-01")
    feb_df = load_raw_csv(RAW_FEB, "2026-02")
    mar_df = load_raw_csv(RAW_MARCH, "2026-03")
    apr_df = load_raw_csv(RAW_APRIL, "2026-04")
    
    df_full = merge_data(dec_df, jan_df, feb_df, mar_df, apr_df)
    
    # ---- 3. 构建特征 ----
    X, ts = build_features(df_full)
    
    # 只保留 2026-03-01 之后的特征，同时保留原始偏移索引
    mask_validation = ts >= np.datetime64('2026-03-01')
    valid_indices = np.where(mask_validation)[0]  # 在完整 X 中的位置，用于标签对齐
    X_val = X[mask_validation]
    ts_val = ts[mask_validation]
    logger.info(f"Validation features: {X_val.shape}, {ts_val[0]} ~ {ts_val[-1]}")
    
    # ---- 4. 预测 + 评估 ----
    results = predict_and_label(X_val, ts_val, df_full, valid_indices)
    evaluator = ModelEvaluator(threshold_up=THRESHOLD_UP, threshold_down=THRESHOLD_DOWN)
    
    # ---- 5. 打印报告 ----
    evaluator.print_report(results['metrics'])
    
    # ---- 6. 保存详情 CSV ----
    save_detailed_csv(results, df_full, EXPERIMENT_DIR, valid_indices)
    
    # ---- 7. 保存评估报告 MD ----
    save_evaluation_report(results['metrics'], EXPERIMENT_DIR)
    
    # ---- 8. 更新主 Excel ----
    exp_id = update_master_xlsx(results['metrics'], EXPERIMENT_DIR)
    
    # ---- 9. 释放内存 ----
    del df_full, feb_df, mar_df, apr_df, X, X_val
    gc.collect()
    
    print("\n" + "=" * 60)
    print(f"  [OK] 验证完成!")
    print(f"  实验 ID:   {exp_id}")
    print(f"  实验目录:  {EXPERIMENT_DIR}")
    print(f"  主记录:    {MASTER_XLSX}")
    print(f"  准确率:    {results['metrics'].get('accuracy', 0):.4f} ({results['metrics'].get('accuracy', 0)*100:.2f}%)")
    print(f"  AUC:       {results['metrics'].get('auc', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
