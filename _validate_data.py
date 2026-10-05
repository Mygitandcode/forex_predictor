"""
_validate_data.py — 校验 E 盘 CSV 数据完整性和连续性
检查项：
  1. 每个 CSV 文件格式是否正确（7列、无空行）
  2. 每月的最后一行与下个月的第一行时间是否衔接（无缺口）
  3. 2026 年数据是否有时间倒序
  4. OHLC 值是否合理（H>=L, O/H/L/C 非负）
"""
import os
import sys
from datetime import datetime, timedelta

CLEANED_ROOT = r"E:\forex_data\cleaned"
PAIRS = ['EURUSD','GBPUSD','USDJPY','AUDUSD','NZDUSD','USDCAD',
         'USDCHF','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY']

# 只检查 2026 年的数据（较新，关注度最高）
CHECK_YEARS = ['2026_04', '2026_05', '2026_06', '2026_07']

results = {}

def parse_row(line: str, pair: str, filename: str, lineno: int) -> dict:
    """解析一行 CSV，返回解析结果或错误"""
    parts = line.strip().split(',')
    if len(parts) != 7:
        return {"error": f"列数={len(parts)}，期望7列", "line": line.strip()}
    
    date_str, time_str, o_str, h_str, l_str, c_str, v_str = parts
    
    # 检查日期格式
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M")
    except ValueError:
        return {"error": f"日期解析失败: {date_str} {time_str}", "line": line.strip()}
    
    # 检查数值
    try:
        o, h, l, c = float(o_str), float(h_str), float(l_str), float(c_str)
        v = int(v_str)
    except ValueError:
        return {"error": f"数值解析失败: O={o_str} H={h_str} L={l_str} C={c_str} V={v_str}", "line": line.strip()}
    
    # 检查 H >= L
    if h < l:
        return {"error": f"H < L: H={h_str} L={l_str}", "line": line.strip()}
    
    # 检查价格非负
    if any(x < 0 for x in [o, h, l, c]):
        return {"error": f"价格小于0: O={o}, H={h}, L={l}, C={c}", "line": line.strip()}
    
    # 检查成交量非负
    if v < 0:
        return {"error": f"成交量小于0: {v}", "line": line.strip()}
    
    return {"ok": True, "dt": dt, "data": (o, h, l, c, v)}


def check_file(filepath: str, pair: str) -> dict:
    """检查单个 CSV 文件"""
    result = {
        "total_rows": 0,
        "errors": [],
        "first_dt": None,
        "last_dt": None,
        "has_gap": False,
        "gap_count": 0,
        "max_gap_minutes": 0,
        "reverse_count": 0,
    }
    
    prev_dt = None
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parsed = parse_row(line, pair, os.path.basename(filepath), lineno)
            result["total_rows"] += 1
            
            if "error" in parsed:
                result["errors"].append(f"  行{lineno}: {parsed['error']}")
                continue
            
            dt = parsed["dt"]
            if result["first_dt"] is None:
                result["first_dt"] = dt
            result["last_dt"] = dt
            
            # 检查连续性
            if prev_dt is not None:
                diff = (dt - prev_dt).total_seconds() / 60
                if diff < 0:
                    result["reverse_count"] += 1
                    result["errors"].append(f"  行{lineno}: 时间倒序 {prev_dt} → {dt}")
                elif diff > 1:
                    gap_minutes = int(diff - 1)
                    result["has_gap"] = True
                    result["gap_count"] += 1
                    result["max_gap_minutes"] = max(result["max_gap_minutes"], gap_minutes)
            
            prev_dt = dt
    
    return result


def check_cross_month(prev_file: str, next_file: str, pair: str) -> dict:
    """检查跨月文件衔接"""
    result = {"has_gap": False, "gap_minutes": 0, "detail": ""}
    
    # 读前一个文件最后一行
    with open(prev_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pass
        last_line = line.strip()
    
    # 读后一个文件第一行
    with open(next_file, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
    
    if not last_line or not first_line:
        return result
    
    last_parts = last_line.split(',')
    first_parts = first_line.split(',')
    
    if len(last_parts) < 2 or len(first_parts) < 2:
        return result
    
    try:
        last_dt = datetime.strptime(f"{last_parts[0]} {last_parts[1]}", "%Y.%m.%d %H:%M")
        first_dt = datetime.strptime(f"{first_parts[0]} {first_parts[1]}", "%Y.%m.%d %H:%M")
        diff = (first_dt - last_dt).total_seconds() / 60
        if diff > 1:
            result["has_gap"] = True
            result["gap_minutes"] = int(diff)
            result["detail"] = f"{last_dt} → {first_dt} 间隔 {int(diff)} 分钟"
    except:
        pass
    
    return result


# ================================================================
# 主程序
# ================================================================
print("=" * 70)
print("E 盘 CSV 数据校验报告")
print("=" * 70)
print()

total_errors = 0
total_gaps = 0
total_reverses = 0

for pair in PAIRS:
    print(f"\n{'─'*70}")
    print(f"  {pair}")
    print(f"{'─'*70}")
    
    pair_dir = os.path.join(CLEANED_ROOT, pair)
    if not os.path.exists(pair_dir):
        print(f"  ❌ 目录不存在")
        continue
    
    months = sorted([d for d in os.listdir(pair_dir) if d.startswith('2026_')])
    if not months:
        print(f"  ⚠️  无 2026 年数据")
        continue
    
    pair_errors = 0
    pair_gaps = 0
    pair_reverses = 0
    file_results = []
    
    for ym in months:
        ym_short = ym.replace("_", "")
        csv_file = os.path.join(pair_dir, ym, f"DAT_MT_{pair}_M1_{ym_short}.csv")
        
        if not os.path.exists(csv_file):
            print(f"  📁 {ym}: 文件不存在")
            continue
        
        # 检查文件大小
        size_kb = os.path.getsize(csv_file) / 1024
        
        # 检查文件
        fr = check_file(csv_file, pair)
        file_results.append((ym, fr, csv_file))
        
        status = "✅" if len(fr["errors"]) == 0 else "❌"
        gap_info = f" | 缺口{fr['gap_count']}处(最大{fr['max_gap_minutes']}min)" if fr["has_gap"] else ""
        rev_info = f" | 倒序{fr['reverse_count']}处" if fr["reverse_count"] > 0 else ""
        print(f"  📁 {ym}: {fr['total_rows']:>8,}行 {size_kb:>8.0f}KB {status}{gap_info}{rev_info}")
        
        for err in fr["errors"][:5]:  # 最多显示5个错误
            print(f"    {err}")
        if len(fr["errors"]) > 5:
            print(f"    ... 还有 {len(fr['errors'])-5} 个错误")
        
        pair_errors += len(fr["errors"])
        pair_gaps += fr["gap_count"]
        pair_reverses += fr["reverse_count"]
    
    # 跨月检查
    prev_file = None
    prev_ym = None
    for ym, fr, csv_file in file_results:
        if prev_file is not None:
            cm = check_cross_month(prev_file, csv_file, pair)
            if cm["has_gap"]:
                print(f"  ⚠️  跨月缺口: {prev_ym}→{ym}: {cm['detail']}")
                pair_gaps += 1
        prev_file = csv_file
        prev_ym = ym
    
    total_errors += pair_errors
    total_gaps += pair_gaps
    total_reverses += pair_reverses

print(f"\n{'='*70}")
print(f"校验汇总")
print(f"{'='*70}")
print(f"  总错误数: {total_errors}")
print(f"  总缺口数: {total_gaps}")
print(f"  总倒序数: {total_reverses}")
print()
if total_errors == 0 and total_gaps == 0 and total_reverses == 0:
    print("  ✅ 所有货币对 2026 年数据格式正确、时间连续、无倒序")
else:
    print("  ⚠️  存在上述问题，请检查")
print(f"{'='*70}")