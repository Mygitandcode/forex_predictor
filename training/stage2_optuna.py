"""
stage2_optuna.py — Stage 2: Optuna + XGBoost 深度调优

纯 XGBoost + Optuna TPE + MLflow + SQLite 断点续训。
每 trial 输出完整的 train/val 指标 + SHAP 特征贡献 Top-10。

目标:
    1. Optuna TPE 采样器搜索 100 次试验 (SQLite 持久化，中断可续)
    2. 每 trial: 80/20 holdout, 双面指标 (train+val AUC/Acc)
    3. 每 trial: SHAP TreeExplainer 输出 Top-10 特征排名
    4. MLflow nested_run 追踪每 trial
    5. 最终: best params 全量训练 → 保存 .pkl → 验证 2026-01~04

使用方式:
    # 逐层验证 (先 1, 再 3, 最后 100):
    python -m training.stage2_optuna --symbol EURUSD --target 15m --n-trials 1
    python -m training.stage2_optuna --symbol EURUSD --target 15m --n-trials 3
    python -m training.stage2_optuna --symbol EURUSD --target 15m --n-trials 100
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_store import FeatureStore
from training.ts_cv import TimeSeriesCV
from training.data_weights import compute_sample_weights
from training.model_eval import ModelEvaluator

warnings.filterwarnings('ignore')
from pipeline.jsonlog import setup_json_logging, jlog  # JSON Lines 日志(替代 basicConfig + print)
logger = logging.getLogger(__name__)

# ---- Monkey-patch: 修复 XGBoost 3.x → SHAP 0.49 base_score 兼容 ----
# XGBoost 3.x 的 save_raw("ubj") 存储 base_score 为 "[0.5]" 数组字符串，
# SHAP 的 XGBTreeModelLoader 调用 float("[0.5]") 会崩溃。
# 通过拦截 decode_ubjson_buffer 在解码后修复 base_score。
def _patch_shap_ubjson():
    """在首次调用前打补丁，幂等。"""
    try:
        import shap.explainers._tree as _tree
        if getattr(_tree, "_forex_patched", False):
            return
        _orig_decode = _tree.decode_ubjson_buffer
        def _patched_decode(fd):
            jmodel = _orig_decode(fd)
            bs = jmodel.get("learner", {}).get("learner_model_param", {}).get("base_score", "")
            if isinstance(bs, str) and bs.startswith("["):
                jmodel["learner"]["learner_model_param"]["base_score"] = str(float(bs.strip("[]")))
            return jmodel
        _tree.decode_ubjson_buffer = _patched_decode
        _tree._forex_patched = True
    except Exception:
        pass  # 静默失败，SHAP fallback 仍会工作

_patch_shap_ubjson()


class Stage2Trainer:
    """Optuna + XGBoost 深度调优训练器

    每 trial 完整输出:
        - train/val AUC + Accuracy (防过拟合)
        - SHAP Top-10 特征贡献排名
        - 信号准确率 (双阈值 0.7/0.3)
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "forex_stage2",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.X_train = None
        self.y_train = None
        self.sample_weights = None
        self.model_type = None

    @staticmethod
    def _read_export_meta(data_dir: str, symbol: str) -> dict:
        """读取 export_meta.json — 训练前的特征集身份证"""
        import json as _json, os as _os
        meta_path = _os.path.join(data_dir, symbol, "export_meta.json")
        if not _os.path.exists(meta_path):
            logger.warning(f"export_meta.json not found at {meta_path}, "
                           f"model_runs will have no selected_features info")
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return _json.load(f)

    @staticmethod
    def _pre_register_run(
        run_id: str,
        symbol: str,
        target_tf: str,
        dimension: int,
        data_dir: str = "E:/forex_npy",
        version_key: str = "unknown",
        run_name: Optional[str] = None,
        study_name: Optional[str] = None,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
    ):
        """训练开始前预注册 model_runs（auc/params 留空，特征集元信息先落库）"""
        import json as _json
        try:
            import pymysql as _pymysql
        except ImportError:
            logger.warning("pymysql not available, skip model_runs pre-registration")
            return

        meta = Stage2Trainer._read_export_meta(data_dir, symbol)
        selected_features = meta.get("selected_features")

        try:
            conn = _pymysql.connect(
                host="localhost", port=3305, user="root", password="",
                database="forex_predictor", charset="utf8mb4",
                connect_timeout=3,
            )
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO model_runs
                    (run_id, run_name, study_name, symbol, target_tf,
                     dimension, n_cols, n_train, n_test, data_dir,
                     feature_version_key, auc, params_json, trials_completed,
                     selected_features)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,%s)
            """, (
                run_id, run_name, study_name, symbol, target_tf,
                dimension, dimension, n_train, n_test, data_dir,
                version_key,
                _json.dumps(selected_features) if selected_features else None,
            ))
            conn.commit()
            cur.close(); conn.close()
            sel_info = f"selected={len(selected_features)}" if selected_features else "all features"
            logger.info(f"model_runs pre-registered: {run_id} | {run_name} | "
                        f"dim={dimension} | ver={version_key} | {sel_info}")
        except Exception as e:
            logger.warning(f"Failed to pre-register model_run: {e}")

    @staticmethod
    def _update_model_run(
        run_id: str,
        auc: float,
        params: dict,
        trials_completed: int,
        model_path: Optional[str] = None,
    ):
        """训练完成后 UPDATE model_runs，填入真实 AUC、超参、模型路径"""
        import json as _json
        try:
            import pymysql as _pymysql
        except ImportError:
            return

        try:
            conn = _pymysql.connect(
                host="localhost", port=3305, user="root", password="",
                database="forex_predictor", charset="utf8mb4",
                connect_timeout=3,
            )
            cur = conn.cursor()
            cur.execute("""
                UPDATE model_runs
                SET auc=%s, params_json=%s, trials_completed=%s, model_path=%s
                WHERE run_id=%s
            """, (auc, _json.dumps(params), trials_completed, model_path, run_id))
            conn.commit()
            cur.close(); conn.close()
            logger.info(f"model_runs updated: {run_id} | auc={auc:.4f} | "
                        f"trials={trials_completed} | path={model_path}")
        except Exception as e:
            logger.warning(f"Failed to update model_run: {e}")

    def run(
        self,
        symbol: str,
        target_window: str = "15m",
        model_type: str = "xgboost",
        data_dir: str = "E:/forex_npy",
        n_trials: int = 100,
        test_size: float = 0.2,
        run_name: Optional[str] = None,
    ) -> Dict:
        """
        执行 Stage 2 Optuna 深度调优 (SQLite 持久化, 中断可续)

        Args:
            symbol: 货币对名称
            target_window: 目标时间窗口
            model_type: xgboost (唯一支持)
            data_dir: 特征存储目录
            n_trials: Optuna 总试验次数 (default 100)
            test_size: 最终测试集比例
            run_name: 实验版本标识 (如 'v2_1060d'), 贯穿 Optuna/MLflow/模型文件/DB

        Returns:
            {best_params, best_cv_auc, study, model, model_path, run_id, trials_completed}

        逐层验证流程:
            1. --n-trials 1    (验证全链路: 数据加载→训练→SHAP→MLflow→pkl)
            2. --n-trials 3    (验证断点续训: 自动跳过已完成的 1 trial)
            3. --n-trials 100  (完整搜索)
        """
        from datetime import datetime as dt
        self.model_type = model_type.lower()
        if run_name is None:
            run_name = dt.now().strftime("%Y%m%d_%H%M")
        
        logger.info(f"=== Stage2: {symbol} @ {target_window}, "
                    f"模型={self.model_type}, 试验={n_trials}, run={run_name} ===")

        # ---- 1. 加载数据 ----
        store = FeatureStore(data_dir)
        data = store.load_all(symbol, mmap_mode='r')  # mmap 避免 16.5 GB 全量加载
        
        X = data['features']
        # load_all returns labels with keys like "15m", "7m", etc. (not "labels_15m")
        y = data.get(target_window)
        if y is None:
            # fallback: try "labels_15m" or bare "labels"
            y = data.get(f'labels_{target_window}', data.get('labels'))
        if y is None:
            available = [k for k in data if not k.startswith('feature') and k != 'timestamps' and k != 'weights']
            raise KeyError(f"Label '{target_window}' not found. Available label keys: {available}")

        # 标签有效性处理：极少数无效行(如 -1) → 钳位到 0，避免 X[valid] 引发 17.7 GB 拷贝
        # X 保持 mmap 只读，XGBoost DMatrix 内拷贝是唯一的内存副本
        n_invalid = int((y < 0).sum())
        if n_invalid > 0:
            if n_invalid < len(y) * 0.001:  # <0.1%，钳位即可
                y = np.clip(y, 0, 1)
                logger.info(f"标签钳位: {n_invalid}/{len(y)} ({n_invalid*100/len(y):.3f}%) 无效行 -> 0")
            else:
                # 大量无效 → 回退到整数索引过滤（会产生拷贝，但比 boolean 拷贝可控）
                valid_idx = np.where(y >= 0)[0]
                X = X[valid_idx]
                y = y[valid_idx]
                if 'timestamps' in data:
                    data['timestamps'] = data['timestamps'][valid_idx]
                logger.info(f"标签过滤: 移除 {n_invalid} 行无效标签")
        
        if 'timestamps' in data:
            timestamps = data['timestamps']
        else:
            timestamps = np.arange(len(y))
        
        logger.info(f"数据: X={X.shape}, 正样本率: {y.sum()/len(y):.1%}")

        # ---- 2. 时序划分 ----
        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X[:split_idx]   # mmap view，不额外占内存
        X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        y_test = y[split_idx:]
        train_ts = timestamps[:split_idx]
        test_ts = timestamps[split_idx:]

        # 计算样本权重
        self.sample_weights = compute_sample_weights(train_ts, lambda_decay=3.0)
        
        logger.info(f"Train: {self.X_train.shape}, Test: {X_test.shape}")

        # ---- 3. Optuna 搜索 (SQLite持久化, 中断不丢) ----
        import mlflow
        import optuna
        from optuna.samplers import TPESampler

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # ---- 读取特征集元信息 (version_key 等) ----
        export_meta = self._read_export_meta(data_dir, symbol)
        version_key = export_meta.get("version_key", "unknown")

        study_name = f"{symbol}_{target_window}_{self.model_type}__{run_name}"
        storage = f"sqlite:///optuna_{study_name}.db"
        
        n_total_target = n_trials  # 目标总 trial 数 (含续跑前已完成)

        # 如果study已存在就继续, 否则创建
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            # 清理僵尸 RUNNING trial + 无效 FAIL (不占 n_trials 额度)
            zombie_count = 0
            for t in list(study.trials):
                if t.state == optuna.trial.TrialState.RUNNING:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                    zombie_count += 1
            completed = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
            # 全是废数据 → 重建
            if completed == 0 and zombie_count > 0:
                logger.warning(f"No completed trials, {zombie_count} zombies -> recreating study")
                optuna.delete_study(study_name=study_name, storage=storage)
                raise KeyError("force recreate")
            # 有已完成 + 有僵尸/失败 -> 多跑 zombie 数量的 trial 补偿
            if zombie_count > 0:
                logger.info(f"Zombie cleanup: {zombie_count} zombies removed, +{zombie_count} compensation trials")
                n_total_target = n_total_target + zombie_count
            # n_trials = 目标总 trial 数; optuna 4.8 optimize(n_trials=N) 是 NEW 语义(N 个新 trial, 非 N total)
            # 续跑只跑差额; completed >= 目标 -> 0 个新 trial (不 over-run 覆盖已有模型, known-issues 2026-08-02)
            n_trials = max(0, n_total_target - completed)
            logger.info(f"续跑已有 study: {completed} completed, target {n_total_target} total -> {n_trials} new"
                        + (f" (cleaned {zombie_count} zombies)" if zombie_count > 0 else ""))
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=10,
                ),
            )
            logger.info(f"新建 study: {study_name}")

        with mlflow.start_run(run_name=study_name) as parent_run:
            mlflow.log_params({
                'symbol': symbol,
                'target_window': target_window,
                'model_type': self.model_type,
                'run_name': run_name,
                'feature_version': version_key,
                'n_trials_target': n_total_target,
                'n_train': len(self.X_train),
                'n_test': len(X_test),
                'n_features': self.X_train.shape[1],
            })

            # ---- 训练前预注册: 特征集元信息 + 物理位置先落库 ----
            self._pre_register_run(
                run_id=parent_run.info.run_id,
                run_name=run_name,
                study_name=study_name,
                symbol=symbol,
                target_tf=target_window,
                dimension=self.X_train.shape[1],
                data_dir=data_dir,
                version_key=version_key,
                n_train=len(self.X_train),
                n_test=len(X_test),
            )

            logger.info(f"开始 Optuna 搜索 (最多 {n_trials} 次试验)...")
            logger.info(f"每 trial 预估 ~10min, 总计约 {n_trials * 10 / 60:.0f}h")
            logger.info(f"Study 存储: {storage} (中断自动续跑)")
            study.optimize(
                self._objective,
                n_trials=n_trials,
                show_progress_bar=True,
            )

            # ---- 4. 最终评估 ----
            best_params = study.best_params
            best_cv_auc = study.best_value
            trials_done = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))

            logger.info(f"\n=== {trials_done}/{n_total_target} trials 完成 ===")
            logger.info(f"Best CV AUC: {best_cv_auc:.4f}")
            logger.info(f"Best Params: {best_params}")

            mlflow.log_metric("best_cv_auc", best_cv_auc)
            mlflow.log_metric("trials_completed", trials_done)

            # 用最佳参数在全量训练集上训练最终模型
            logger.info("Training final model with best params...")
            best_params.update({'n_jobs': 1, 'random_state': 42, 'verbosity': 1})
            from xgboost import XGBClassifier
            final_model = XGBClassifier(**{k:v for k,v in best_params.items() 
                                           if k in ['n_estimators','max_depth','learning_rate',
                                                   'subsample','colsample_bytree','min_child_weight',
                                                   'reg_alpha','reg_lambda','gamma',
                                                   'n_jobs','random_state','verbosity']},
                                           early_stopping_rounds=100)
            final_model.fit(self.X_train, self.y_train,
                            eval_set=[(X_test, y_test)],
                            verbose=False)
            
            # 保存模型 + 特征快照
            model_path = f"./models/{study_name}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import joblib
            joblib.dump(final_model, model_path)
            # 特征集快照 — 与模型同名，方便回溯
            meta_snapshot = os.path.join(os.path.dirname(model_path), f"{study_name}_meta.json")
            import json as _json
            with open(meta_snapshot, "w", encoding="utf-8") as f:
                _json.dump(export_meta, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(meta_snapshot)
            logger.info(f"Model saved: {model_path} (+ {meta_snapshot})")

            # ---- 5. 2026-01~04 真实验证 ----
            # 注: 已由 pipeline stage5_validate.py 替代，此处仅作为兼容保留
            # 超时或失败时不影响模型保存
            logger.info("\n=== 2026-01~04 真实验证 (可选) ===")
            try:
                import subprocess
                validate_script = str(Path(__file__).parent.parent / "_validate_0326.py")
                result_v = subprocess.run(
                    [sys.executable, validate_script],
                    capture_output=True, text=True, timeout=300
                )
                if result_v.stdout:
                    logger.info(result_v.stdout[:500])
                if result_v.returncode != 0 and result_v.stderr:
                    logger.warning(f"Validate stderr: {result_v.stderr[:500]}")
            except subprocess.TimeoutExpired:
                logger.warning("_validate_0326.py 超时, 已跳过 (模型已保存)")
            except Exception as e:
                logger.warning(f"_validate_0326.py 失败: {e}, 已跳过 (模型已保存)")

            result = {
                'study': study,
                'best_params': best_params,
                'best_cv_auc': best_cv_auc,
                'trials_completed': trials_done,
                'model': final_model,
                'model_path': model_path,
                'run_id': parent_run.info.run_id,
            }

            # ---- 6. 训练后更新 model_runs: 填入真实 AUC/params/模型路径 ----
            self._update_model_run(
                run_id=parent_run.info.run_id,
                auc=best_cv_auc,
                params=best_params,
                trials_completed=trials_done,
                model_path=os.path.abspath(model_path),
            )

            logger.info(f"\nDONE | {trials_done} trials | Best AUC={best_cv_auc:.4f} | Model={model_path}")
        return result

    def _objective(self, trial) -> float:
        """Optuna objective — 每 trial 完整评估 (train/val/SHAP)

        输出:
            - train AUC + Accuracy    (训练集, 看是否过拟合)
            - val AUC + Accuracy      (验证集, 优化目标)
            - signal Acc + Ratio      (双阈值 0.7/0.3)
            - SHAP Top-10 特征排名    (贡献率追踪)
            - MLflow nested run
        """
        import optuna, time as _time, mlflow, gc
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from datetime import datetime

        t_start = _time.time()

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 1,  # avoid Windows multi-thread deadlock
        }

        # ---- 80/20 时间切分 (holdout, 不做CV以保持每个trial可对比) ----
        split = int(len(self.X_train) * 0.8)
        X_tr, X_val = self.X_train[:split], self.X_train[split:]
        y_tr, y_val = self.y_train[:split], self.y_train[split:]

        # ---- 训练 ----
        logger.info(
            f"Trial {trial.number:3d} | Training start: "
            f"n_est={params['n_estimators']} lr={params['learning_rate']:.4f} "
            f"depth={params['max_depth']} | "
            f"X_tr={X_tr.shape} X_val={X_val.shape}"
        )
        params_with_es = {**params, 'early_stopping_rounds': 100}
        model = XGBClassifier(**params_with_es)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(f"Trial {trial.number:3d} | Training done, evaluating...")

        # ---- Train 指标 (检测过拟合) ----
        y_tr_prob = model.predict_proba(X_tr)[:, 1]
        train_auc = roc_auc_score(y_tr, y_tr_prob)
        y_tr_pred = (y_tr_prob >= 0.5).astype(int)
        train_acc = accuracy_score(y_tr, y_tr_pred)

        # ---- Val 指标 (优化目标) ----
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        y_val_pred_hard = (y_val_prob >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, y_val_pred_hard)

        # ---- Gap (过拟合温度计, backlog #4 第一要务) ----
        auc_gap = train_auc - val_auc

        # 双阈值信号准确率
        y_val_signal = np.full_like(y_val_prob, -1, dtype=int)
        y_val_signal[y_val_prob >= 0.7] = 1
        y_val_signal[y_val_prob <= 0.3] = 0
        has_signal = y_val_signal >= 0
        sig_acc = (
            (y_val_signal[has_signal] == y_val[has_signal]).sum() / max(has_signal.sum(), 1)
            if has_signal.sum() > 0 else 0.0
        )
        sig_ratio = has_signal.sum() / len(y_val)

        # ---- 分类指标 (成绩单 backlog #1, y_val 与 0.5 阈值向量 L511 在手) ----
        val_precision = precision_score(y_val, y_val_pred_hard, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred_hard, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred_hard, zero_division=0)
        # KS: 按 prob 排序累计正/负样本占比的最大间距 (~5 行)
        _order = np.argsort(y_val_prob)
        _y_s = y_val[_order]
        _n_pos = max(float(_y_s.sum()), 1.0)
        _n_neg = max(len(_y_s) - float(_y_s.sum()), 1.0)
        _cdf_pos = np.cumsum(_y_s) / _n_pos
        _cdf_neg = np.cumsum(1 - _y_s) / _n_neg
        val_ks = float(np.max(np.abs(_cdf_pos - _cdf_neg))) if len(_y_s) > 0 else 0.0
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_hard, labels=[0, 1]).ravel()
        val_confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        # ---- SHAP 特征贡献 Top-10 ----
        top_features_str = ""
        try:
            import shap
            _patch_shap_ubjson()  # 确保 monkey-patch 已生效
            explainer = shap.TreeExplainer(model.get_booster())
            shap_vals = explainer.shap_values(X_val[:min(10000, len(X_val))])
            shap_importance = np.abs(shap_vals).mean(axis=0)
            top10_idx = np.argsort(shap_importance)[-10:][::-1]
            top_features_list = [
                f"f{i}({shap_importance[i]:.4f})" for i in top10_idx
            ]
            top_features_str = ", ".join(top_features_list)
        except Exception as e:
            logger.warning(f"  SHAP failed: {e}")
            top_features_str = "SHAP_FAILED"

        elapsed = _time.time() - t_start
        n_est_used = model.best_iteration or params['n_estimators']

        logger.info(
            f"Trial {trial.number:3d} | "
            f"TrA={train_auc:.4f} TrAcc={train_acc:.3f} | "
            f"VaA={val_auc:.4f} VaAcc={val_acc:.3f} | "
            f"Gap={auc_gap:.4f}{' ⚠过拟合' if auc_gap > 0.02 else ''} | "
            f"SigAcc={sig_acc:.3f} Sig%={sig_ratio:.2%} | "
            f"lr={params['learning_rate']:.4f} dp={params['max_depth']} "
            f"tree={n_est_used} | "
            f"{elapsed:.0f}s | "
            f"TOP10: {top_features_str[:120]}{'...' if len(top_features_str) > 120 else ''}"
        )

        # ---- MLflow nested run (每 trial 独立记录) ----
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(params)
            mlflow.log_metrics({
                'train_auc': train_auc,
                'train_acc': train_acc,
                'val_auc': val_auc,
                'auc_gap': auc_gap,
                'val_acc': val_acc,
                'val_signal_acc': sig_acc,
                'val_signal_ratio': sig_ratio,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_ks': val_ks,
                'confusion_tn': val_confusion['tn'],
                'confusion_fp': val_confusion['fp'],
                'confusion_fn': val_confusion['fn'],
                'confusion_tp': val_confusion['tp'],
                'trial_number': trial.number,
                'duration_s': round(elapsed, 1),
                'n_estimators_used': int(n_est_used),
            })
            mlflow.set_tag("shap_top10", top_features_str)

        # ---- user_attrs 10 项 (07 成绩单直读源; 必须在 pruning 判定前——PRUNED trial 亦落库) ----
        trial.set_user_attr('train_auc', float(train_auc))
        trial.set_user_attr('val_auc', float(val_auc))
        trial.set_user_attr('auc_gap', float(auc_gap))
        trial.set_user_attr('precision', float(val_precision))
        trial.set_user_attr('recall', float(val_recall))
        trial.set_user_attr('f1', float(val_f1))
        trial.set_user_attr('ks', val_ks)
        trial.set_user_attr('confusion', val_confusion)
        trial.set_user_attr('signal_acc', float(sig_acc))
        trial.set_user_attr('signal_ratio', float(sig_ratio))

        # ---- Optuna Pruning (基于 val_auc) ----
        trial.report(val_auc, 1)
        if trial.should_prune():
            # 显式释放大对象再让 Optuna 剪枝
            del model
            gc.collect()
            raise optuna.TrialPruned()

        # 每 trial 结束强制 GC，避免 100+ trial 内存累积
        del model
        gc.collect()
        return val_auc


# ---- CLI 入口 ----
if __name__ == "__main__":
    setup_json_logging()

    parser = argparse.ArgumentParser(description="Stage2: Optuna + XGBoost 深度调优")
    parser.add_argument("--symbol", required=True, help="货币对名称")
    parser.add_argument("--target", default="15m", help="目标时间窗口")
    parser.add_argument("--model-type", default="xgboost",
                        choices=["xgboost"], help="模型类型 (仅 xgboost)")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna 总试验次数")
    parser.add_argument("--data-dir", default="E:/forex_npy", help="特征存储目录")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow 地址")
    parser.add_argument("--test-size", type=float, default=0.2, help="最终测试集比例")
    parser.add_argument("--run-name", default=None,
                        help="实验版本标识 (如 v2_1060d), 贯穿 Optuna/MLflow/模型文件")
    args = parser.parse_args()

    trainer = Stage2Trainer(tracking_uri=args.mlflow_uri)

    result = trainer.run(
        symbol=args.symbol,
        target_window=args.target,
        model_type=args.model_type,
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        test_size=args.test_size,
        run_name=args.run_name,
    )

    jlog("result", best_params=result['best_params'], best_cv_auc=round(result['best_cv_auc'], 4),
         run_id=result['run_id'], trials_completed=result['trials_completed'])
