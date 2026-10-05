# Forex Predictor — MT4 多货币对涨跌预测训练系统

对 MT4 平台 12 个货币对（EURUSD、GBPUSD、USDJPY 等）的 1 分钟 K 线进行实时分析，预测未来 7m/10m/15m/30m/1h 后的涨跌方向。

## 技术栈

Python 3.10 ｜ XGBoost ｜ Optuna（超参搜索）｜ MLflow（实验追踪）｜ SHAP（特征可解释性）｜ MariaDB 10.11 ｜ FastAPI ｜ Streamlit ｜ Docker Compose ｜ Redis（MQ 驱动）

## 系统架构

```
数据接入 → 特征工程 → 训练管线 → 服务层 → 可视化
```

| 层 | 模块 | 职责 |
|----|------|------|
| 数据接入 | `data/` | MT4 HST/CSV 解析、Dukascopy tick 下载、MySQL 同步 |
| 特征工程 | `features/` | 多时间框架联合特征（1m/5m/15m/30m/1h × 5 层 ≈ 1060 维）|
| 训练管线 | `pipeline/` | MQ（Redis）驱动的 6-Stage 全链路：拉齐数据→特征构建→导出→训练→验证→报告 |
| 训练引擎 | `training/` | XGBoost + Optuna 两阶段训练、时间衰减权重、评估标准 |
| 服务层 | `serving/` + `api/` | FastAPI 推理服务（端口 8000）、MySQL HTTP 代理、MLflow 追踪 |
| 可视化 | `dashboard/` | Streamlit 仪表盘（端口 8501）：K 线、训练概览、信号与 SHAP |

## 训练管线设计要点

- **幂等预检**：每个 Stage 执行前检查数据对齐状态，已完成步骤直接跳过，支持断点续跑
- **多币种参数化**：symbol / target_tf 全链参数化，特征表按币种段独立重建（互不覆盖）
- **双阈值评估**：0.7/0.3 阈值信号体系 + AUC/gap/P/R/F1/KS/混淆矩阵全套指标
- **时间序纪律**：训练/验证按时间序切分（holdout 后 4 自然月），标签前瞻窗口与特征口径一致，防时序泄露
- **多币种共存**：candles/features 全表 `(timestamp, symbol)` 联合主键，JOIN 双表币种绑定

## 快速开始

```powershell
# 环境
conda / venv Python 3.10.9
pip install -r requirements.txt

# 依赖服务：MariaDB 10.11 (localhost:3305)、Redis (6379)、MLflow (5001)

# 数据同步（默认 EURUSD）
python -u _data_sync.py --symbol EURUSD

# 特征构建 + 导出
python -u _build_features_v2.py --mode full --symbol EURUSD
python -u _export_v2.py --symbol EURUSD --dry-run

# 训练（Optuna 40 trials）
python -m training.stage2_optuna --symbol EURUSD --target 15m --n-trials 40

# 服务
python -m uvicorn serving.fastapi_app:app --port 8000
python -m streamlit run dashboard/streamlit_app.py --server.port 8501
```

## 说明

- 本仓库收录**代码空间**（源码/配置/文档/工具脚本）；venv、模型产物（.pkl）、MLflow/optuna 实验状态、分析产物均在 `.gitignore`，克隆后按需重建
- 历史行情数据与模型不在仓库内（本地 E 盘管理）
- 项目开发过程：基于 HistData/MT4/Dukascopy 数据，两阶段 XGBoost 训练 + 特征迭代自治，AI 协作开发

> 作者：维克兹 ｜ 本仓库同时作为 Python/ML 后端面试作品集（开发记录见提交历史）
