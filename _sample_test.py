"""Test: MT 2024, ASCII 2024 for true M1 bars. Fixed MT parsing."""
import requests, re, zipfile, pandas as pd
from pathlib import Path

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "http://www.histdata.com/",
})

def download_analyze(platform, year, pair):
    plat_lower = "ascii" if platform == "ASCII" else "metatrader"
    page_url = f"http://www.histdata.com/download-free-forex-historical-data/?/{plat_lower}/1-minute-bar-quotes/{pair.lower()}/{year}"
    r = session.get(page_url, timeout=15)
    tk_match = re.search(r'name="tk"[^>]+value="([^"]+)"', r.text)
    if not tk_match: return
    r2 = session.post("http://www.histdata.com/get.php", data={
        "tk": tk_match.group(1), "date": year, "datemonth": year,
        "platform": platform, "timeframe": "M1", "fxpair": pair,
    }, timeout=120)
    if r2.status_code != 200 or len(r2.content) < 5000: return
    
    out_dir = Path(r"E:\forex_data\cleaned") / pair / year
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"test_{platform}_{year}.zip"
    with open(zip_path, "wb") as f:
        f.write(r2.content)
    
    csv_path = None
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if n.endswith('.csv'):
                zf.extract(n, out_dir)
                csv_path = out_dir / n
    
    # MT format: "2000.05.30,17:27,0.9302,..." (date, time, OHLCV = 7 cols, comma)
    # ASCII: "20000530 172700;0.9302;..." (datetime, OHLCV = 6 cols, semicolon)
    with open(csv_path) as f:
        first = f.readline()
    
    if ';' in first:
        # ASCII: 6 columns
        df = pd.read_csv(csv_path, sep=';', header=None,
                         names=['dt_str', 'open', 'high', 'low', 'close', 'volume'])
        df['dt'] = pd.to_datetime(df['dt_str'].astype(str), format='%Y%m%d %H%M%S', errors='coerce')
    else:
        # MT: 7 columns (date, time, o, h, l, c, v)
        df = pd.read_csv(csv_path, sep=',', header=None,
                         names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        df['dt'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            format='%Y.%m.%d %H:%M', errors='coerce'
        )
    
    df = df.sort_values('dt').dropna(subset=['dt', 'open', 'high', 'low', 'close'])
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    gaps = df['dt'].diff().dropna().dt.total_seconds()
    
    at_min = (df['dt'].dt.second == 0).mean() * 100
    gap60 = (gaps == 60).mean() * 100
    
    print(f"  [{platform:5s} {year}] {len(df):>7,} rows | {df['dt'].min().date()} -> {df['dt'].max().date()}")
    print(f"          sec=00: {at_min:5.1f}% | gap=60s: {gap60:5.1f}% | gap_med: {gaps.median():6.0f}s")
    print(f"          price: {df['close'].min():.5f} ~ {df['close'].max():.5f}")
    
    if gap60 >= 95:
        print(f"          ===> TRUE M1 !!!")
    else:
        print(f"          ===> tick data")

print("Downloading & analyzing...\n")

for platform, year in [("MT","2024"), ("ASCII","2024")]:
    try:
        download_analyze(platform, year, "EURUSD")
    except Exception as e:
        print(f"  [{platform} {year}] ERROR: {e}")
