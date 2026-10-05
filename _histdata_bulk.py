"""
Full run: 12 pairs, reverse chronological, smart format selection.
Logs to E:\forex_data\cleaned\download_log.md
Active data to E:\forex_data\active\{pair}\{year}\
"""
import requests, re, zipfile, pandas as pd, shutil, time
from pathlib import Path
from datetime import datetime

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD",
         "USDCHF","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY"]
START_YEAR, END_YEAR = 2024, 2000
CLEANED = Path(r"E:\forex_data\cleaned")
ACTIVE = Path(r"E:\forex_data\active")
LOG = CLEANED / "download_log.md"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "http://www.histdata.com/",
})

def log(msg):
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def dl(pair, year, fmt):
    plat = "ascii" if fmt == "ASCII" else "metatrader"
    url = (f"http://www.histdata.com/download-free-forex-historical-data/"
           f"?/{plat}/1-minute-bar-quotes/{pair.lower()}/{year}")
    try:
        r = session.get(url, timeout=15)
    except:
        return None
    tk = re.search(r'name="tk"[^>]+value="([^"]+)"', r.text)
    if not tk: return None
    try:
        r2 = session.post("http://www.histdata.com/get.php", data={
            "tk": tk.group(1), "date": str(year), "datemonth": str(year),
            "platform": fmt, "timeframe": "M1", "fxpair": pair,
        }, timeout=120)
    except:
        return None
    if r2.status_code != 200 or len(r2.content) < 5000: return None
    
    out = CLEANED / pair / str(year)
    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"HISTDATA_COM_{fmt}_{pair}_M1_{year}.zip"
    try:
        zp.write_bytes(r2.content)
    except:
        return None
    
    try:
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if n.endswith('.csv'):
                    zf.extract(n, out)
                    return out / n
    except:
        return None
    return None

def val(csv_path):
    try:
        with open(csv_path) as f:
            first = f.readline()
        sep = ';' if ';' in first else ','
        if sep == ',':
            df = pd.read_csv(csv_path, sep=',', header=None,
                             names=['date','time','o','h','l','c','v'])
            df['dt'] = pd.to_datetime(df['date'].astype(str)+' '+df['time'].astype(str),
                                      format='%Y.%m.%d %H:%M', errors='coerce')
        else:
            df = pd.read_csv(csv_path, sep=';', header=None,
                             names=['s','o','h','l','c','v'])
            df['dt'] = pd.to_datetime(df['s'].astype(str), format='%Y%m%d %H%M%S', errors='coerce')
        df = df.dropna(subset=['dt','o','h','l','c'])
        if len(df) < 100: return (False, 0, None, None, 0, "")
        df[['o','h','l','c']] = df[['o','h','l','c']].astype(float)
        df = df.sort_values('dt')
        gaps = df['dt'].diff().dropna().dt.total_seconds()
        gap60 = (gaps == 60).mean() * 100
        pr = f"{df['c'].min():.5f}~{df['c'].max():.5f}"
        return (gap60 >= 95, len(df), df['dt'].min(), df['dt'].max(), gap60, pr)
    except:
        return (False, 0, None, None, 0, "error")

def copy_active(pair, year, csv):
    dest = ACTIVE / pair / str(year)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv, dest / f"{pair}_M1_{year}.csv")

# === RUN ===
with open(LOG, "w", encoding="utf-8") as f:
    f.write(f"# Histdata Download Log\n**Started**: {datetime.now():%Y-%m-%d %H:%M}\n")
    f.write(f"**Pairs**: {', '.join(PAIRS)}\n\n")

for pi, pair in enumerate(PAIRS, 1):
    log(f"\n## {pair} ({pi}/{len(PAIRS)})")
    log(f"| Year | Fmt | M1? | Rows | Date Range | gap=60s | Price |")
    log(f"|------|-----|-----|------|------------|---------|-------|")
    
    fmt = "MT"
    bad = 0
    valid = 0
    
    for yr in range(START_YEAR, END_YEAR - 1, -1):
        csv = dl(pair, yr, fmt)
        if not csv:
            alt = "ASCII" if fmt == "MT" else "MT"
            log(f"  [{yr}] {fmt} failed, trying {alt}...")
            csv = dl(pair, yr, alt)
            if csv:
                fmt = alt
            else:
                log(f"| {yr} | - | FAIL | - | - | - | download failed |")
                bad += 1
                if bad >= 3:
                    log(f"> Stop at {yr}: 3 consecutive years unavailable.")
                    break
                continue
        
        ok, rows, ds, de, g60, pr = val(csv)
        ds_s = ds.strftime("%Y-%m-%d") if ds else "-"
        de_s = de.strftime("%Y-%m-%d") if de else "-"
        label = "YES" if ok else "NO(tick)"
        log(f"| {yr} | {fmt:4s} | {label:7s} | {rows:,} | {ds_s} -> {de_s} | {g60:.1f}% | {pr} |")
        
        if ok:
            copy_active(pair, yr, csv)
            valid += 1
            bad = 0
        else:
            log(f"> Stopped at {yr}: tick data.")
            break
        
        time.sleep(1)
    
    log(f"> {pair}: {valid} valid M1 years downloaded.")

log(f"\n---\n**Finished**: {datetime.now():%Y-%m-%d %H:%M}")
print("\nAll done!")
