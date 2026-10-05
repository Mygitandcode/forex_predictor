"""
pipeline/stage6_report.py — Stage 6: 报告聚合

聚合全链路产物 → outputs/{YYYY_MM}/{trailName}/
  - 02_feature_importance.md    (模型 gain 排名)
  - 06_summary.md               (训练总览)
  - 07_training_summary.md      (Trial 全表/过拟合诊断/结论 — 2026-09-13 新增, 成绩单 backlog #1+backlog5)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np

from .mq import StageRunner, StageResult
from .feature_selector import get_effective_features

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Stage6Report(StageRunner):
    name = "报告聚合"
    stage_num = 6

    def run(self, config: dict, prev_data: dict) -> StageResult:
        trail_name = config["trailName"]
        output_dir = prev_data.get("output_dir", "")
        model_path = prev_data.get("model_path", "")
        mode = config.get("mode", "train")

        # ---- 1. 特征重要性报告 ----
        imp_path = os.path.join(output_dir, "02_feature_importance.md")
        if mode == "train" or not os.path.exists(imp_path):
            self._gen_feature_importance(imp_path, model_path, config)
            logger.info(f"特征重要性报告: {imp_path}")

        # ---- 2. 训练总览 ----
        summary_path = os.path.join(output_dir, "06_summary.md")
        self._gen_summary(summary_path, config, prev_data, model_path, mode)
        logger.info(f"训练总览: {summary_path}")

        # ---- 3. 训练收尾报告 07 (读不到 study 时降级 warning, 不 fail Stage6) ----
        try:
            training_summary_path = os.path.join(output_dir, "07_training_summary.md")
            self._gen_training_summary(training_summary_path, config)
            logger.info(f"训练收尾报告: {training_summary_path}")
        except Exception as e:  # noqa: BLE001 — Stage6 是链路末端，fail 会挡 downstream 完工
            logger.warning(f"07_training_summary 生成失败(Stage6 继续): {e}")

        return StageResult(ok=True, stage=6,
                           message="报告聚合完成",
                           data={"output_dir": output_dir})

    def _gen_feature_importance(self, path: str, model_path: str, config: dict):
        """生成特征重要性排名 Markdown"""
        features = config.get("features", [])
        trail_name = config["trailName"]

        lines = [
            f"# 特征重要性排名 — {trail_name}",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 特征总数: {len(features)}",
            "",
        ]

        # 读取模型 gain
        if model_path and os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                booster = model.get_booster()
                gain = booster.get_score(importance_type='gain')

                # 构建 (name, gain) 列表并排序
                ranked = sorted(gain.items(), key=lambda x: x[1], reverse=True)

                lines.append("## 完整排名")
                lines.append("")
                lines.append("| Rank | Feature | Gain |")
                lines.append("|------|---------|------|")

                for rank, (fname, val) in enumerate(ranked, 1):
                    lines.append(f"| {rank} | `{fname}` | {val:.1f} |")

                # 统计
                n_zero = sum(1 for _, v in ranked if v == 0)
                lines.extend([
                    "",
                    "## 统计",
                    f"- 有贡献特征: {len(ranked) - n_zero}",
                    f"- 零贡献特征: {n_zero}",
                    f"- 总特征: {len(ranked)}",
                ])
            except Exception as e:
                lines.append(f"> ⚠️ 无法读取模型 gain: {e}")
        else:
            lines.append("> 模型文件不存在，无法生成特征重要性")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _gen_summary(self, path: str, config: dict, prev_data: dict, model_path: str, mode: str):
        """生成训练总览"""
        lines = [
            f"# 训练总览 — {config['trailName']}",
            "",
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **货币对**: {config.get('symbol', 'EURUSD')}",
            f"- **目标周期**: {config.get('target_tf', '15m')}",
            f"- **特征数**: {len(config.get('features', []))}",
            f"- **Trials**: {config.get('n_trials', 0)}",
            f"- **Mode**: {mode}",
            f"- **Model**: {os.path.basename(model_path) if model_path else 'N/A'}",
            "",
            "## 产出文件",
            f"- `01_input_config.md`",
            f"- `02_feature_importance.md`",
            f"- `03_live_predictions.xlsx`",
            f"- `05_model.pkl`",
            f"- `06_summary.md` (本文件)",
        ]

        if os.path.exists(path):
            # 追加模式
            lines.insert(0, "")
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # ---- 07_training_summary（成绩单，2026-09-13 新增）----

    def _load_study_for_config(self, config: dict):
        """study 定位公式: {symbol}_{target_tf}_xgboost__{trailName}；db=项目根绝对路径。
        model_type 硬编码 xgboost（schema 无 model_type 属性；本计划只跑 xgboost）。"""
        import optuna

        symbol = config.get('symbol', 'EURUSD')
        target_tf = config.get('target_tf', '15m')
        trail_name = config['trailName']
        study_name = f"{symbol}_{target_tf}_xgboost__{trail_name}"
        db_path = os.path.join(_PROJECT_ROOT, f"optuna_{study_name}.db")
        return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

    def _mlflow_run_metrics(self, study_name: str):
        """本地 file store(mlruns/) 兜底扫描: {trial_number: {metric: value(取末行)}}。
        生产 5001 服务端时本地无数据→空 dict（各列如实"—"，不造假源）。"""
        import re as _re

        runs_root = os.path.join(_PROJECT_ROOT, "mlruns")
        if not os.path.isdir(runs_root):
            return {}
        runs = []
        for dirpath, dirnames, _ in os.walk(runs_root):
            if os.path.basename(dirpath) != "tags":
                continue
            rdir = os.path.dirname(dirpath)
            tags = {}
            for tf in os.listdir(dirpath):
                try:
                    with open(os.path.join(dirpath, tf), encoding="utf-8", errors="replace") as f:
                        tags[tf] = f.read().strip()
                except OSError:
                    continue
            runs.append((os.path.getmtime(rdir), rdir, tags))
        runs.sort(key=lambda x: x[0], reverse=True)  # 同名 parent 取最新
        parent_ids = {os.path.basename(d) for _, d, t in runs if t.get("mlflow.runName") == study_name}
        out = {}
        for _, rdir, tags in runs:
            if tags.get("mlflow.parentRunId") not in parent_ids:
                continue
            m = _re.match(r"trial_(\d{3})$", tags.get("mlflow.runName", ""))
            if not m:
                continue
            bucket = out.setdefault(int(m.group(1)), {})
            mdir = os.path.join(rdir, "metrics")
            if not os.path.isdir(mdir):
                continue
            for metric in os.listdir(mdir):
                try:
                    with open(os.path.join(mdir, metric), encoding="utf-8") as f:
                        last = ""
                        for line in f:
                            if line.strip():
                                last = line
                    bucket[metric] = float(last.split()[1])
                except (OSError, IndexError, ValueError):
                    continue
        return out

    def _gen_training_summary(self, path: str, config: dict, study=None):
        """07 训练收尾报告。独立可调用（A2-04 单元验证路径）: study 可显式传入；
        缺省按 config 定位公式加载。覆盖写（全量重算的整表，追加会重复）。"""
        trail_name = config["trailName"]
        if study is None:
            study = self._load_study_for_config(config)
        trials = sorted(study.trials, key=lambda t: t.number)
        mlflow_map = self._mlflow_run_metrics(study.study_name)

        def _src(t, key):
            a = t.user_attrs or {}
            if a.get(key) is not None:
                return a[key]
            m = mlflow_map.get(t.number, {})
            return m.get(key) if m else None

        def _fmt(v, nd=4):
            return (f"{v:.{nd}f}" if isinstance(v, (int, float)) and v is not None else "—")

        rows, gaps = [], []
        for t in trials:
            tr_a, va_a = _src(t, 'train_auc'), _src(t, 'val_auc')
            gap = _src(t, 'auc_gap')
            if gap is None and tr_a is not None and va_a is not None:
                gap = tr_a - va_a  # 差值现算=两值均在手时唯一合法（"现算"语义）
            conf = (t.user_attrs or {}).get('confusion') or mlflow_map.get(t.number, {})
            kv = {k: conf.get(k) if isinstance(conf, dict) else None
                  for k in ('tp', 'fp', 'tn', 'fn')}
            sig = _src(t, 'signal_ratio')
            gap_txt = _fmt(gap) + (" ⚠" if gap is not None and float(gap) > 0.02 else "")
            if gap is not None:
                gaps.append((t.number, float(gap), t.state.name))
            rows.append("| {n} | {st} | {tr} | {va} | {gp} | {pr} | {re} | {f1} | {ks} | {tp} | {fp} | {tn} | {fn} | {sr} |".format(
                n=t.number, st=t.state.name, tr=_fmt(tr_a), va=_fmt(va_a), gp=gap_txt,
                pr=_fmt(_src(t, 'precision')), re=_fmt(_src(t, 'recall')), f1=_fmt(_src(t, 'f1')),
                ks=_fmt(_src(t, 'ks')), tp=_fmt(kv['tp'], 0), fp=_fmt(kv['fp'], 0),
                tn=_fmt(kv['tn'], 0), fn=_fmt(kv['fn'], 0), sr=(f"{sig:.4f}" if isinstance(sig, (int, float)) else "—")))

        lines = [f"# 训练收尾报告 — {trail_name}", "",
                 f"> 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"研究: {study.study_name}（{len(trials)} trials）",
                 "> 口径: gap=train_auc−val_auc（取值优先级 user_attrs→MLflow→两者皆无='—'，禁止 0 填充）；P/R/F1/KS/混淆对无源旧 trial 为'—'"]
        lines += ["", "## 一、Trial 全表", "",
                  "| # | state | train_auc | val_auc | gap | precision | recall | F1 | KS | TP | FP | TN | FN | 信号率 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows

        comp = [g for g in gaps if g[2] != 'PRUNED']
        pruned = [g for g in gaps if g[2] == 'PRUNED']
        lines += ["", "## 二、过拟合诊断小结"]
        best_auc = None
        if comp:
            best_aucs = [(t.number, _src(t, 'val_auc')) for t in trials if t.state.name == 'COMPLETE' and _src(t, 'val_auc') is not None]
            if best_aucs:
                bnum, best_auc = max(best_aucs, key=lambda x: x[1])
            worst = max(comp, key=lambda g: g[1])
            lines += [
                f"- best trial: #{bnum}（val_auc={best_auc:.4f}）" if best_auc is not None else "- best trial: —（COMPLETE 皆无 val_auc 源）",
                f"- gap 均值: {sum(g[1] for g in comp) / len(comp):.4f}（统计样本=COMPLETE {len(comp)} 个；未计 PRUNED {len(pruned)} 个）",
                f"- gap>0.02 trial 数: {sum(1 for g in comp if g[1] > 0.02)}",
                f"- 最过拟合: trial #{worst[0]} gap={worst[1]:.4f}",
            ]
        else:
            lines += ["- 无 COMPLETE trial（best/best gap/诊断不可得——纯推理格或 study 读取异常）"]

        lines += ["", "## 三、结论行"]
        if comp:
            max_gap = max(g[1] for g in comp)
            verdict = "健康" if max_gap <= 0.02 else ("轻度过拟合" if max_gap <= 0.08 else "严重过拟合")
            best_gap_now = min(g[1] for g in comp)
            auc_txt = f"{best_auc:.4f}" if best_auc is not None else "—"
            lines.append(f"本次训练 best AUC={auc_txt}，best gap={best_gap_now:.4f}，判定={verdict}")
        else:
            lines.append("结论: 无 COMPLETE trial，无法给出 AUC/gap 判定（07 记录为空壳）")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
