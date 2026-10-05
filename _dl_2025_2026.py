"""Download 2025 and 2026 (year + monthly) for all 12 pairs."""
import requests, re, zipfile, pandas as pd, shutil, time
from pathlib import Path

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD",
         "USDCHF","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY"]
CLEANED = Path(r"E:\forex_data\cleaned")
ACTIVE = Path(r"E:\forex_data\active")

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "http://www.histdata.com/"})

def dl(pair, year, month=None):
    plat = "metatrader"
    if month:
        url = (f"http://www.histdata.com/download-free-forex-historical-data/"
               f"?/{plat}/1-minute-bar-quotes/{pair.lower()}/{year}/{month}")
    else:
        url = (f"http://www.histdata.com/download-free-forex-historical-data/"
               f"?/{plat}/1-minute-bar-quotes/{pair.lower()}/{year}")
    try:
        r = session.get(url, timeout=15)
    except:
        return None
    tk = re.search(r'name="tk"[^>]+value="([^"]+)"', r.text)
    if not tk: return None
    payload = {
        "tk": tk.group(1), "date": str(year),
        "datemonth": str(month) if month else str(year),
        "platform": "MT", "timeframe": "M1", "fxpair": pair,
    }
    try:
        r2 = session.post("http://www.histdata.com/get.php", data=payload, timeout=120)
    except:
        return None
    if r2.status_code != 200 or len(r2.content) < 5000: return None
    
    sub = str(year) if not month else f"{year}_{month:02d}"
    out = CLEANED / pair / sub
    out.mkdir(parents=True, exist_ok=True)
    label = f"{pair}_M1_{year}" if not month else f"{pair}_M1_{year}_{month:02d}"
    zp = out / f"{label}.zip"
    zp.write_bytes(r2.content)
    
    with zipfile.ZipFile(zp) as zf:
        for n in zf.namelist():
            if n.endswith('.csv'):
                zf.extract(n, out)
                csv = out / n
                # Validate & copy to active
                with open(csv) as f:
                    first = f.readline()
                sep = ';' if ';' in first else ','
                df = pd.read_csv(csv, sep=sep, header=None)
                active_sub = str(year) if not month else f"{year}_{month:02d}"
                dest = ACTIVE / pair / active_sub
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv, dest / f"{pair}_M1_{active_sub}.csv")
                return (len(df), csv)
    return None

print("Downloading 2025 & 2026...")
results = []
for pair in PAIRS:
    # 2025 full year
    r = dl(pair, 2025)
    results.append((pair, "2025", r[0] if r else "FAIL"))
    print(f"  {pair:8s} 2025: {r[0] if r else 'FAIL'} rows")
    
    # 2026 monthly
    for month in range(1, 13):
        r = dl(pair, 2026, month)
        status = f"{r[0]} rows" if r else "FAIL"
        results.append((pair, f"2026-{month:02d}", status))
        print(f"  {pair:8s} 2026-{month:02d}: {status}")
    time.sleep(0.5)

print("\n=== Summary ===")
print(f"{'Pair':<12} {'Period':<12} {'Rows'}")
for p, period, rows in results:
    print(f"{p:<12} {period:<12} {rows}")
