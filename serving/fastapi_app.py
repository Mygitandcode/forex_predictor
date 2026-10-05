"""
fastapi_app.py — FastAPI 推理服务

提供 REST API 接口：
- POST /predict         单次预测
- POST /predict/batch   批量预测
- GET  /health          健康检查
- GET  /models          列出已加载模型
- GET  /api/analysis    分析接口（Phase 4）

使用方式:
    uvicorn serving.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from serving.mysql_proxy import router as mysql_router, init_pool as mysql_init_pool

# Phase 4 API routes
from api.candles import router as candles_router
from api.predictions import router as predictions_router
from api.shap_query import router as shap_router
from api.runs import router as runs_router
from api.feature_versions import router as feature_versions_router
from api.assessment import router as assessment_router
from api.adopt import router as adopt_router

logger = logging.getLogger(__name__)

# ---- FastAPI App ----
app = FastAPI(
    title="MT4 Forex Predictor API",
    description="多货币对预测推理服务",
    version="1.0.0",
)

# 挂载 MySQL 代理路由
app.include_router(mysql_router)

# Phase 4 API routes
app.include_router(candles_router)
app.include_router(predictions_router)
app.include_router(shap_router)
app.include_router(runs_router)
app.include_router(feature_versions_router)
app.include_router(assessment_router)
app.include_router(adopt_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 数据模型 ----
class KLineData(BaseModel):
    """单条 K 线数据"""
    symbol: str = Field(..., example="EURUSD")
    timestamp: str = Field(..., example="2024-01-15 09:01:00")
    open: float
    high: float
    low: float
    close: float
    tick_volume: Optional[int] = 0
    spread: Optional[int] = 0


class PredictRequest(BaseModel):
    """预测请求"""
    kline: KLineData
    history: Optional[List[KLineData]] = None  # 历史K线（可选）


class PredictResponse(BaseModel):
    """预测响应"""
    symbol: str
    timestamp: str
    predictions: Dict[str, float]  # {"7m": 0.65, "10m": 0.72, ...}
    signals: Dict[str, str]        # {"7m": "震荡", "10m": "涨", ...}
    inference_time_ms: float
    model_version: Optional[str] = None


class BatchPredictRequest(BaseModel):
    """批量预测请求"""
    klines: List[KLineData]


class BatchPredictResponse(BaseModel):
    """批量预测响应"""
    results: List[PredictResponse]


# ---- 全局状态 ----
loaded_models: Dict[str, Any] = {}  # {symbol: model}
history_buffers: Dict[str, List[Dict]] = {}  # {symbol: [kline, ...]}
MODEL_DIR = Path("./models")


def load_models(symbols: Optional[List[str]] = None):
    """
    加载训练好的模型
    
    模型文件命名: {symbol}_{target}.json
    如: EURUSD_15m.json
    """
    global loaded_models
    
    if not MODEL_DIR.exists():
        logger.warning(f"模型目录不存在: {MODEL_DIR}")
        return

    for model_file in MODEL_DIR.glob("*.json"):
        name = model_file.stem
        parts = name.split("_")
        if len(parts) >= 2:
            symbol = parts[0]
        else:
            symbol = name

        if symbols and symbol not in symbols:
            continue

        try:
            import joblib
            # 尝试用 mlflow 加载
            # model = mlflow.pyfunc.load_model(model_file)
            model = joblib.load(model_file)
            loaded_models[symbol] = model
            logger.info(f"已加载模型: {symbol} ← {model_file}")
        except Exception as e:
            logger.error(f"加载模型失败 {model_file}: {e}")

    # 也加载 .pkl 文件
    for model_file in MODEL_DIR.glob("*.pkl"):
        name = model_file.stem
        parts = name.split("_")
        symbol = parts[0] if parts else name

        if symbol in loaded_models:
            continue
        if symbols and symbol not in symbols:
            continue

        try:
            import joblib
            model = joblib.load(model_file)
            loaded_models[symbol] = model
            logger.info(f"已加载模型: {symbol} ← {model_file}")
        except Exception as e:
            logger.error(f"加载模型失败 {model_file}: {e}")


def predict_single(
    symbol: str,
    features: np.ndarray,
) -> Dict[str, float]:
    """
    单次推理
    
    Args:
        symbol: 货币对名称
        features: (1, D) 特征向量
        
    Returns:
        {window: probability} 每个时间窗口的涨概率
    """
    model = loaded_models.get(symbol)
    if model is None:
        raise ValueError(f"模型未加载: {symbol}")

    # 预测
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)
        if isinstance(proba, list):
            proba = proba[0]
        if hasattr(proba, 'ndim') and proba.ndim == 2:
            proba = proba[:, 1]  # 涨概率
    else:
        proba = model.predict(features)

    # 映射到时间窗口
    windows = ["7m", "10m", "15m", "30m", "60m"]
    
    if hasattr(proba, 'flatten'):
        proba = proba.flatten()
    
    predictions = {}
    for i, w in enumerate(windows):
        if i < len(proba):
            predictions[w] = float(proba[i])
        else:
            predictions[w] = float(proba[0])

    return predictions


def classify_signal(prob: float, up_thresh: float = 0.7, down_thresh: float = 0.3) -> str:
    """双阈值分类"""
    if prob >= up_thresh:
        return "涨"
    elif prob <= down_thresh:
        return "跌"
    else:
        return "震荡"


# ---- API 路由 ----

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(loaded_models),
        "symbols": list(loaded_models.keys()),
    }


@app.get("/models")
async def list_models():
    """列出已加载的模型"""
    return {
        "models": {
            symbol: str(type(model).__name__)
            for symbol, model in loaded_models.items()
        }
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    单次预测
    
    接收当前 K 线和可选历史数据，返回各时间窗口的预测结果。
    """
    start_time = datetime.now()
    
    kline = request.kline
    symbol = kline.symbol

    # 检查模型是否可用
    if symbol not in loaded_models:
        # 尝试动态加载
        if not loaded_models:
            load_models()
        if symbol not in loaded_models:
            raise HTTPException(
                status_code=503,
                detail=f"模型未就绪: {symbol}。请先完成训练并放置模型到 {MODEL_DIR}"
            )

    try:
        # TODO: 这里需要调用 FeatureBuilder.build()
        # 目前作为占位，使用简单的特征
        # 实际使用时，需要维护每个 symbol 的历史缓冲区
        
        # 管理历史缓冲区
        if symbol not in history_buffers:
            history_buffers[symbol] = []
        history_buffers[symbol].append(kline.dict())
        
        # 保留最近 500 条
        if len(history_buffers[symbol]) > 500:
            history_buffers[symbol] = history_buffers[symbol][-500:]

        # 构建特征向量（需要完整实现）
        # features = feature_builder.build(kline.dict(), history_df)
        # 占位：使用 K 线数值作为简单特征
        features = np.array([[
            kline.open, kline.high, kline.low, kline.close,
            kline.high - max(kline.open, kline.close),
            min(kline.open, kline.close) - kline.low,
            abs(kline.close - kline.open),
        ]], dtype=np.float32)

        # 推理
        predictions = predict_single(symbol, features)
        
        # 分类信号
        signals = {
            w: classify_signal(p)
            for w, p in predictions.items()
        }

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return PredictResponse(
            symbol=symbol,
            timestamp=kline.timestamp,
            predictions=predictions,
            signals=signals,
            inference_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """批量预测"""
    results = []
    for kline in request.klines:
        single_request = PredictRequest(kline=kline)
        result = await predict(single_request)
        results.append(result)

    return BatchPredictResponse(results=results)


@app.get("/api/analysis/{symbol}")
async def analysis(symbol: str):
    """
    分析接口 (Phase 4)
    
    返回当前向量与聚类中心的位置关系、SHAP 值等
    """
    return {
        "symbol": symbol,
        "status": "not_implemented",
        "message": "分析接口将在 Phase 4 实现",
    }


# ---- 启动事件 ----
@app.on_event("startup")
async def startup_event():
    """服务启动时自动加载模型 + 初始化 MySQL 连接池"""
    logger.info("FastAPI 推理服务启动中...")
    
    # 初始化 MySQL 代理
    try:
        mysql_init_pool()
    except Exception as e:
        logger.warning(f"MySQL 代理初始化失败（XAMPP 可能未启动）: {e}")
    
    # 自动加载 models 目录下的模型
    if MODEL_DIR.exists():
        load_models()
    
    if loaded_models:
        logger.info(f"已加载模型: {list(loaded_models.keys())}")
    else:
        logger.warning(
            f"未找到模型文件。请将训练好的模型放入 {MODEL_DIR.absolute()}"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理"""
    logger.info("推理服务已关闭")


# ---- 直接运行 ----
if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    uvicorn.run(
        "serving.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
