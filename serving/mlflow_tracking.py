"""
mlflow_tracking.py — MLflow 实验记录与模型管理

功能:
    1. 启动本地 MLflow Server
    2. 实验管理（创建/切换实验）
    3. 模型注册（Registry: Staging → Production → Archived）
    4. 模型加载与推理
    5. 实验对比查询

使用方式:
    # 启动 MLflow Server
    python -m serving.mlflow_tracking serve
    
    # 注册模型
    python -m serving.mlflow_tracking register --run-id <id> --name EURUSD_15m
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class MLflowManager:
    """MLflow 实验追踪与模型管理"""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "forex_predictor",
        artifact_root: str = "./mlruns",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = artifact_root

    def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        backend_store: str = "./mlruns",
    ):
        """启动 MLflow Tracking Server（本地）"""
        cmd = [
            "mlflow", "server",
            "--host", host,
            "--port", str(port),
            "--backend-store-uri", f"file:///{os.path.abspath(backend_store)}",
            "--default-artifact-root", f"file:///{os.path.abspath(self.artifact_root)}",
        ]
        
        logger.info(f"启动 MLflow Server: {' '.join(cmd)}")
        logger.info(f"访问 MLflow UI: http://localhost:{port}")
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            logger.info("MLflow Server 已停止")
        except FileNotFoundError:
            logger.error(
                "mlflow 命令不存在。请先安装: pip install mlflow"
            )

    def get_experiment(self):
        """获取或创建实验"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"创建实验: {self.experiment_name} (id={experiment_id})")
        else:
            logger.info(f"使用已有实验: {self.experiment_name} (id={experiment.experiment_id})")
        
        return experiment

    def list_runs(
        self,
        order_by: str = "metrics.test_accuracy DESC",
        max_results: int = 20,
    ) -> List[Dict]:
        """列出实验中的运行记录"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        experiment = self.get_experiment()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[order_by],
            max_results=max_results,
        )
        return runs.to_dict('records')

    def get_best_run(
        self, metric: str = "metrics.test_accuracy"
    ) -> Optional[Dict]:
        """获取最佳运行的模型"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        runs = mlflow.search_runs(
            experiment_ids=[self.get_experiment().experiment_id],
            order_by=[f"{metric} DESC"],
            max_results=1,
        )
        
        if len(runs) == 0:
            return None
        
        best = runs.iloc[0]
        return {
            'run_id': best.run_id,
            'metrics': {k: v for k, v in best.items() if k.startswith('metrics.')},
            'params': {k: v for k, v in best.items() if k.startswith('params.')},
        }

    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "Staging",
    ):
        """注册模型到 MLflow Model Registry"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            # 获取 run 的模型 URI
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"模型已注册: {model_name} v{result.version}")
            
            # 转换 Stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage,
            )
            logger.info(f"模型 Stage: {stage}")
            
            return result
        except Exception as e:
            logger.error(f"模型注册失败: {e}")
            return None

    def promote_model(
        self,
        model_name: str,
        version: int,
        to_stage: str = "Production",
    ):
        """提升模型阶段 (Staging → Production)"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=to_stage,
        )
        logger.info(f"模型 {model_name} v{version} → {to_stage}")

    def load_production_model(self, model_name: str):
        """加载 Production 阶段模型"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            logger.info(f"已加载 Production 模型: {model_name}")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None

    def compare_models(self, metric: str = "test_accuracy"):
        """对比实验中所有模型的指标"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        runs = mlflow.search_runs(
            experiment_ids=[self.get_experiment().experiment_id],
        )
        
        if len(runs) == 0:
            print("暂无实验记录")
            return None
        
        metric_cols = [c for c in runs.columns if c.startswith('metrics.')]
        param_cols = [c for c in runs.columns if c.startswith('params.symbol') or c.startswith('params.target_window')]
        
        display_cols = ['run_id'] + param_cols + metric_cols
        display_cols = [c for c in display_cols if c in runs.columns]
        
        comparison = runs[display_cols].sort_values(
            f'metrics.{metric}', ascending=False
        )
        
        print(f"\n模型对比 (按 {metric} 排序):")
        print(comparison.to_string())
        
        return comparison


# ---- CLI ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="MLflow 实验管理工具")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # serve
    serve_parser = subparsers.add_parser('serve', help='启动 MLflow Server')
    serve_parser.add_argument('--host', default='0.0.0.0')
    serve_parser.add_argument('--port', type=int, default=5000)

    # list
    list_parser = subparsers.add_parser('list', help='列出实验运行')

    # best
    best_parser = subparsers.add_parser('best', help='查看最佳模型')

    # register
    reg_parser = subparsers.add_parser('register', help='注册模型')
    reg_parser.add_argument('--run-id', required=True)
    reg_parser.add_argument('--name', required=True)
    reg_parser.add_argument('--stage', default='Staging')

    # compare
    cmp_parser = subparsers.add_parser('compare', help='对比模型')

    args = parser.parse_args()
    mgr = MLflowManager()

    if args.command == 'serve':
        mgr.start_server(host=args.host, port=args.port)
    elif args.command == 'list':
        runs = mgr.list_runs()
        for r in runs[:10]:
            print(f"  {r.get('run_id', '?')[:8]}... "
                  f"accuracy={r.get('metrics.test_accuracy', 'N/A')}")
    elif args.command == 'best':
        best = mgr.get_best_run()
        if best:
            print(f"最佳 Run: {best['run_id']}")
            for k, v in best['metrics'].items():
                print(f"  {k}: {v}")
        else:
            print("暂无实验记录")
    elif args.command == 'register':
        mgr.register_model(args.run_id, args.name, args.stage)
    elif args.command == 'compare':
        mgr.compare_models()
    else:
        parser.print_help()
