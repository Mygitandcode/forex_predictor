"""
mt4_connector.py — ZeroMQ Subscriber，接收 MT4 实时 K 线数据

从 MT4 平台的 MQL4 EA 脚本通过 ZeroMQ PUB 推送 1分钟 K 线，
Python 端通过 zmq.SUB 阻塞接收，实现事件驱动、低延迟数据获取。

使用方式:
    from data.mt4_connector import MT4Connector
    conn = MT4Connector()
    
    # 阻塞接收模式
    for k_data in conn.receive_stream():
        print(f"收到 {k_data['symbol']} K线: {k_data['close']}")
    
    # 异步单条接收
    k_data = conn.receive_one(timeout_ms=5000)
"""

import zmq
import json
import time
import logging
from typing import Dict, Optional, Generator, List

logger = logging.getLogger(__name__)


# ---- K线数据格式 ----
# MT4 推送的 JSON 格式:
# {
#     "symbol": "EURUSD",
#     "timestamp": "2024-01-15 09:01:00",  # K线起始时间 (UTC)
#     "open": 1.09523,
#     "high": 1.09545,
#     "low": 1.09510,
#     "close": 1.09530,
#     "tick_volume": 156,
#     "spread": 12
# }


class MT4Connector:
    """ZeroMQ SUB 端，接收 MT4 实时 K 线推送"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        symbols: Optional[List[str]] = None,
    ):
        """
        初始化 ZeroMQ 订阅连接
        
        Args:
            host: MT4 所在主机 IP（本地为 127.0.0.1）
            port: ZeroMQ PUB 端口
            symbols: 要订阅的货币对列表，None 表示订阅全部
        """
        self.host = host
        self.port = port
        self.symbols = symbols or []
        self.address = f"tcp://{host}:{port}"
        
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        
        # 设置订阅过滤
        if self.symbols:
            for s in self.symbols:
                self._socket.setsockopt_string(zmq.SUBSCRIBE, s)
                logger.info(f"订阅货币对: {s}")
        else:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            logger.info("订阅全部消息")

        self._connected = False

    def connect(self):
        """建立 ZeroMQ 连接"""
        self._socket.connect(self.address)
        self._connected = True
        logger.info(f"已连接到 MT4 ZeroMQ PUB: {self.address}")
        return self

    def disconnect(self):
        """断开 ZeroMQ 连接"""
        if self._connected:
            self._socket.close()
            self._ctx.term()
            self._connected = False
            logger.info("已断开 ZeroMQ 连接")

    def receive_one(self, timeout_ms: int = 5000) -> Optional[Dict]:
        """
        接收一条 K 线数据（带超时）
        
        Args:
            timeout_ms: 超时时间（毫秒），None 表示无限等待
            
        Returns:
            解析后的 K 线字典，超时返回 None
        """
        if not self._connected:
            self.connect()

        # 设置接收超时
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        try:
            raw = self._socket.recv_string()
            # MT4 推送格式: "SYMBOL {json}"
            # 去掉前缀拿到 JSON 部分
            parts = raw.split(" ", 1)
            if len(parts) == 2:
                symbol, json_str = parts
            else:
                json_str = parts[0]
                symbol = "UNKNOWN"

            k_data = json.loads(json_str)
            if "symbol" not in k_data:
                k_data["symbol"] = symbol

            logger.debug(f"收到 K 线: {k_data['symbol']} @ {k_data.get('timestamp', '?')}")
            return k_data

        except zmq.Again:
            logger.debug("接收超时")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}, 原始数据: {raw}")
            return None

    def receive_stream(self) -> Generator[Dict, None, None]:
        """
        阻塞接收流，生成器模式
        
        Yields:
            每条 K 线数据字典
        """
        if not self._connected:
            self.connect()

        logger.info("开始接收实时 K 线流...")
        while True:
            try:
                k_data = self.receive_one(timeout_ms=-1)  # 无限等待
                if k_data:
                    yield k_data
            except KeyboardInterrupt:
                logger.info("用户中断，停止接收")
                break
            except Exception as e:
                logger.error(f"接收异常: {e}")
                time.sleep(1)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ---- 快速测试入口 ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # 测试: 连接本地 MT4，接收 EURUSD
    conn = MT4Connector(symbols=["EURUSD"])
    try:
        with conn:
            for i, k_data in enumerate(conn.receive_stream()):
                print(f"[{i}] {k_data['symbol']}: "
                      f"O={k_data['open']:.5f} H={k_data['high']:.5f} "
                      f"L={k_data['low']:.5f} C={k_data['close']:.5f}")
                if i >= 10:
                    break
    except KeyboardInterrupt:
        print("用户中断")
