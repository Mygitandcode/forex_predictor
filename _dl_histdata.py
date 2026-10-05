"""
Download EURUSD data for one year from histdata.com (Generic ASCII format).
"""
import requests, re, os, zipfile
from pathlib import Path

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "http://www.histdata.com/",
})

year = "2000"
pair = "EURUSD"

# 1. GET ASCII page to extract tk
page_url = f"http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"
print(f"[1] GET: {page_url}")
r = session.get(page_url, timeout=15)
r.raise_for_status()

tk_match = re.search(r'name="tk"\s+id="tk"\s+value="([^"]+)"', r.text)
if not tk_match:
    print("FAIL: no tk token")
    exit(1)
tk = tk_match.group(1)
print(f"[1] tk={tk}")

# 2. POST download
payload = {
    "tk": tk,
    "date": year,
    "datemonth": year,
    "platform": "ASCII",
    "timeframe": "M1",
    "fxpair": pair,
}
print(f"[2] POST /get.php")
r = session.post("http://www.histdata.com/get.php", data=payload, timeout=120)
print(f"[2] Status={r.status_code}, Content-Type={r.headers.get('Content-Type','?')}, Size={len(r.content)}")

# 3. Save
out_dir = Path(r"E:\forex_data\cleaned") / pair / year
out_dir.mkdir(parents=True, exist_ok=True)

is_html = b"<!doctype" in r.content[:200].lower() or b"<html" in r.content[:200].lower()
if is_html or len(r.content) < 100:
    print("[2] ERROR: Got HTML or tiny response (not a ZIP). Saving response.html")
    with open(out_dir / "_response.html", "wb") as f:
        f.write(r.content)
    exit(1)

zip_path = out_dir / f"HISTDATA_COM_ASCII_{pair}_M1_{year}.zip"
with open(zip_path, "wb") as f:
    f.write(r.content)
print(f"[3] Saved: {zip_path} ({len(r.content):,} bytes)")

# 4. Check ZIP
try:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"[4] ZIP: {len(names)} files")
        for n in names[:5]:
            info = zf.getinfo(n)
            print(f"     {n} ({info.file_size:,} bytes)")
        
        # Extract
        for n in names:
            zf.extract(n, out_dir)
            print(f"     Extracted: {n}")
except zipfile.BadZipFile:
    print("[4] ERROR: Invalid ZIP. First bytes:", r.content[:100])
