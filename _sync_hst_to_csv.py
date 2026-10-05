"""
_sync_hst_to_csv.py — 从 MT4 HST 导出数据补齐 E 盘 CSV
将 HST 中 CSV 缺失的数据（2026-04-10 之后）按每月分割导出到 E:\forex_data\cleaned\{pair}\{YYYY_MM}\
数据格式与已有 CSV 一致：DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME（无表头，逗号分隔）
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from data.hst_importer import HSTImporter

# === 配置 ===
CLEANED_ROOT = r"E:\forex_data\cleaned"
HST_DIR = r"C:\Users\admin\AppData\Roaming\MetaQuotes\Terminal\7F9720261EB20B849E02C5EEE9135BF9\history\FXOAK-Real01"

# 12 个货币对
PAIRS = ['EURUSD','GBPUSD','USDJPY','AUDUSD','NZDUSD','USDCAD',
         'USDCHF','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY']

# CSV 最后日期（从实际数据读取）
CSV_LAST_DATE = "2026-04-10"

imp = HSTImporter(hst_dir=HST_DIR, output_dir=r"E:\forex_data\mt4_hst")

def export_hst_to_csv(pair):
    """将 HST 中 CSV 缺口部分导出到 CSV 文件"""
    print(f"\n{'='*60}")
    print(f"处理 {pair}...")
    
    # 1. 读 HST 数据
    df_hst = imp.import_symbol(pair)
    if df_hst is None or len(df_hst) == 0:
        print(f"  ❌ HST 无数据，跳过")
        return 0
    
    # 2. 过滤出 CSV 缺口部分（> 2026-04-10）
    cutoff = pd.Timestamp("2026-04-11")
    df_new = df_hst[df_hst.index >= cutoff].copy()
    
    if len(df_new) == 0:
        print(f"  ⏭️  无新数据需要补齐")
        return 0
    
    print(f"  HST 总行数: {len(df_hst):,}")
    print(f"  需补齐行数: {len(df_new):,}")
    print(f"  范围: {df_new.index.min()} ~ {df_new.index.max()}")
    
    # 3. 准备 CSV 格式数据
    records = []
    for idx, row in df_new.iterrows():
        # idx 是 datetime 类型
        dt_str = idx.strftime("%Y.%m.%d")
        tm_str = idx.strftime("%H:%M")
        o = f"{row['open']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['open']:.5f}"
        h = f"{row['high']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['high']:.5f}"
        l = f"{row['low']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['low']:.5f}"
        c = f"{row['close']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['close']:.5f}"
        v = int(row['volume']) if 'volume' in row else 0
        records.append(f"{dt_str},{tm_str},{o},{h},{l},{c},{v}")
    
    # 4. 按年月分组，写入对应文件
    month_groups = {}
    for idx, row in df_new.iterrows():
        ym = idx.strftime("%Y_%m")
        if ym not in month_groups:
            month_groups[ym] = []
        dt_str = idx.strftime("%Y.%m.%d")
        tm_str = idx.strftime("%H:%M")
        o = f"{row['open']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['open']:.5f}"
        h = f"{row['high']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['high']:.5f}"
        l = f"{row['low']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['low']:.5f}"
        c = f"{row['close']:.6f}" if pair in ('USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY') else f"{row['close']:.5f}"
        v = int(row['volume']) if 'volume' in row else 0
        month_groups[ym].append(f"{dt_str},{tm_str},{o},{h},{l},{c},{v}")
    
    total_written = 0
    for ym, lines in sorted(month_groups.items()):
        ym_short = ym.replace("_", "")  # 2026_04 -> 202604
        out_dir = os.path.join(CLEANED_ROOT, pair, ym)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"DAT_MT_{pair}_M1_{ym_short}.csv")
        
        # 检查文件是否存在，存在则追加
        mode = 'a' if os.path.exists(out_file) else 'w'
        with open(out_file, mode, encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
        
        written = len(lines)
        total_written += written
        print(f"  📁 {ym}: {out_file} ({written:,} 行, {'追加' if mode=='a' else '新建'})")
    
    print(f"  ✅ {pair} 补齐完成，共 {total_written:,} 行")
    return total_written


# === 主程序 ===
print("=" * 60)
print("MT4 HST → E 盘 CSV 数据补齐")
print("=" * 60)
print(f"CSV 最新数据截止: {CSV_LAST_DATE}")
print(f"HST 数据源: {HST_DIR}")
print(f"输出目录: {CLEANED_ROOT}")
print()

total_all = 0
for pair in PAIRS:
    n = export_hst_to_csv(pair)
    total_all += n

print(f"\n{'='*60}")
print(f"全部完成！共补齐 {total_all:,} 行数据")
print(f"{'='*60}")