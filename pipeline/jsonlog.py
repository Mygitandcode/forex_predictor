"""pipeline/jsonlog.py - JSON Lines 日志工具

所有训练管线日志统一用 JSON Lines 格式(每行一个 JSON 对象, 向后追加)。
无状态 AI 读日志末尾若干行即可反序列化恢复状态, 无需正则解析文本。

铁律(guide §三): 所有能用 JSON Lines 的日志都用 JSON Lines, 一律向后追加。

提供:
  - JsonFormatter   : logging.Formatter 子类, 把 LogRecord 格式化成 JSON 行
  - jlog(event,**k) : 便捷函数, 输出一条 JSON 行到 stdout(给 print 脚本用)
  - setup_json_logging(log_path) : 配置 root logger 用 JsonFormatter(stdout + 可选文件)

设计:
  - orchestrator.py 用 setup_json_logging(log_path) -> root logger 写 JSON Lines 到文件+stdout
  - stage2_optuna.py 用 setup_json_logging() -> stdout(JSON Lines), 由 stage4 重定向到日志文件
  - _export_v2.py 用 jlog() -> stdout(JSON Lines), 由 stage3 重定向到日志文件
  - 这样 E:/forex_logs/{trailName}.log 全是 JSON Lines(orchestrator FileHandler + 子脚本 stdout 混写, 行级原子)
"""

import json
import logging
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化成 JSON Lines(一行一个 JSON 对象)。

    输出格式:
        {"ts":"2026-08-02T11:47:50","level":"INFO","logger":"__main__","msg":"收到请求"}
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def jlog(event: str, **kwargs):
    """便捷函数: 输出一条 JSON Lines 到 stdout。

    给 _export_v2.py 等纯 print 脚本用, 替代 print(f"...")。
    每次调用输出一行: {"ts":"...","event":"...","key":"value",...}

    Args:
        event: 事件类型(如 "progress"/"done"/"error")
        **kwargs: 附加字段(值需 JSON 可序列化; tuple/shape 转 list)

    示例:
        jlog("progress", written=20000, total=4186899, pct=0, last_ts="2015-01-21")
        jlog("done", status="dims_correct")
        jlog("error", msg="--features JSON must be a list")
    """
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event}
    entry.update(kwargs)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def setup_json_logging(log_path: str = None, level: int = logging.INFO):
    """配置 root logger 用 JsonFormatter(stdout + 可选文件)。

    清除现有 handler 后重新配置(basicConfig 残留的文本 formatter 会被清除)。

    Args:
        log_path: 若给定, 额外写 JSON Lines 到此文件(向后追加, mode='a')。
        level: 日志级别(默认 INFO)。
    """
    root = logging.getLogger()
    root.setLevel(level)
    # 清除现有 handler(避免 basicConfig 残留的文本 formatter)
    for h in list(root.handlers):
        root.removeHandler(h)
    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(JsonFormatter())
    root.addHandler(sh)
    # file handler(可选)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
