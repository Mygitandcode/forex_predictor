"""Step 2b: Find correct Generic ASCII download URL for histdata.com."""
import requests, re

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# The user selected "Generic ASCII" format. Let's check different URL patterns.
base = "http://www.histdata.com/download-free-forex-historical-data/?/"

patterns = [
    "metatrader/1-minute-bar-quotes/eurusd/2000",       # MT format (failed)
    "ascii/1-minute-bar-quotes/eurusd/2000",             # maybe?
    "generic/1-minute-bar-quotes/eurusd/2000",           # maybe?
]

for p in patterns:
    url = base + p
    try:
        r = session.get(url, timeout=10, allow_redirects=True)
        has_form = '/get.php' in r.text
        has_tk = 'name="tk"' in r.text
        title_match = re.search(r'<title>([^<]+)</title>', r.text)
        title = title_match.group(1) if title_match else "N/A"
        print(f"[{r.status_code}] {p}")
        print(f"   Title: {title[:80]}")
        print(f"   Has get.php: {has_form}, Has tk: {has_tk}, Size: {len(r.text)}")
        print()
    except Exception as e:
        print(f"FAIL: {p} -> {e}")
        print()
