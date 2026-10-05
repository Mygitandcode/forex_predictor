// ============================================================
// zmq_bridge.mq4 — MT4 EA 脚本，通过 ZeroMQ 推送 1分钟 K 线
// ============================================================
// 
// 功能: 每根新 1分钟 K 线触发，将 OHLC 数据通过 ZeroMQ PUB 推送给 Python
// 
// 安装步骤:
//   1. 将本文件放入 MT4 的 Experts 目录
//      (通常: C:\Users\<用户名>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL4\Experts\)
//   2. 将 ZMQ DLL 库文件放入 MQL4\Libraries\
//      下载: https://github.com/dingmaotu/mql-zmq
//      文件: Zmq.mqh, Zmq.lib, libzmq.dll
//   3. 在 MT4 中编译 (F7) 本脚本
//   4. 拖拽到图表上运行
//
// 依赖: mql-zmq (https://github.com/dingmaotu/mql-zmq)
//   - 下载 Zmq.mqh 放入 MQL4\Include\Zmq\
//   - 下载 Zmq.lib 放入 MQL4\Libraries\
//   - libzmq.dll 放入 MT4 根目录或 C:\Windows\System32\
//
// 货币对配置: 修改下方的 SYMBOLS 数组
// ZeroMQ 端口: 修改 PORT 常量 (默认 5555)
// ============================================================

#property copyright "forex_predictor"
#property link      ""
#property version   "1.00"
#property strict

// ---- 引入 ZeroMQ 库 ----
// 安装 mql-zmq 后取消下面注释:
// #include <Zmq\Zmq.mqh>

// ---- 配置区 ----
#define PORT 5555
#define ADDRESS "tcp://*:5555"

// 要推送的货币对列表
string SYMBOLS[] = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF"
};

// ============================================================
// 注意: 以下代码需在安装 mql-zmq 后使用
// 
// 完整 EA 代码框架:
// ============================================================

/*
// 全局变量
SocketContext g_ctx;
Socket g_pub("");  // PUB socket

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // 创建 ZeroMQ 上下文和 PUB socket
    // g_ctx = SocketContext::Create();
    // g_pub = g_ctx.CreateSocket(ZMQ_PUB);
    // g_pub.Bind(ADDRESS);
    
    Print("ZeroMQ PUB 已启动: ", ADDRESS);
    Print("推送货币对: ");
    for(int i = 0; i < ArraySize(SYMBOLS); i++)
        Print("  ", SYMBOLS[i]);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // g_pub.Close();
    // g_ctx.Shutdown();
    Print("ZeroMQ PUB 已关闭");
}

//+------------------------------------------------------------------+
//| Expert tick function — 每 tick 都触发                              |
//| 但只在新的 1分钟 K 线形成时推送数据                                    |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime last_bar_time = 0;
    
    // 检查所有配置的货币对
    for(int i = 0; i < ArraySize(SYMBOLS); i++)
    {
        string symbol = SYMBOLS[i];
        
        // 获取当前 1分钟 K 线时间
        datetime bar_time = iTime(symbol, PERIOD_M1, 0);
        
        // 如果是新 K 线
        if(bar_time != last_bar_time)
        {
            last_bar_time = bar_time;
            
            // 获取 OHLC 数据
            double open  = iOpen(symbol, PERIOD_M1, 0);
            double high  = iHigh(symbol, PERIOD_M1, 0);
            double low   = iLow(symbol, PERIOD_M1, 0);
            double close = iClose(symbol, PERIOD_M1, 0);
            long volume  = iVolume(symbol, PERIOD_M1, 0);
            int spread   = (int)MarketInfo(symbol, MODE_SPREAD);
            
            // 构造 JSON 消息
            string json = StringFormat(
                "{\"symbol\":\"%s\",\"timestamp\":\"%s\","
                "\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,"
                "\"tick_volume\":%d,\"spread\":%d}",
                symbol,
                TimeToString(bar_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                open, high, low, close,
                volume, spread
            );
            
            // 发送: 格式 = "SYMBOL {json}" (方便 Python 端过滤)
            string message = symbol + " " + json;
            
            // g_pub.Send(message);
            
            Print("推送: ", symbol, " @ ", TimeToString(bar_time));
        }
    }
}
*/

//+------------------------------------------------------------------+
// 简易版脚本 (无需 ZeroMQ，写 CSV 测试)
// 在安装 ZeroMQ 库之前，可先用此脚本验证数据获取逻辑
//+------------------------------------------------------------------+
void OnStart()
{
    Print("=== MT4 K线数据导出测试 ===");
    
    string symbol = "EURUSD";
    int bars = 10;
    
    for(int i = bars - 1; i >= 0; i--)
    {
        datetime t = iTime(symbol, PERIOD_M1, i);
        double o = iOpen(symbol, PERIOD_M1, i);
        double h = iHigh(symbol, PERIOD_M1, i);
        double l = iLow(symbol, PERIOD_M1, i);
        double c = iClose(symbol, PERIOD_M1, i);
        
        PrintFormat("[%s] O=%.5f H=%.5f L=%.5f C=%.5f",
            TimeToString(t), o, h, l, c);
    }
}
