"""Step 1: Probe histdata.com page structure to find download URL."""
import requests, re, sys

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# Fetch the EURUSD 2000 M1 data page
url = "http://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/eurusd/2000"
try:
    r = session.get(url, timeout=15)
    print(f"Status: {r.status_code}, Size: {len(r.text)} chars")
    
    # Find all links/hrefs
    links = re.findall(r'href=[\'"]?([^\'" >]+)', r.text)
    forms = re.findall(r'<form[^>]*action=[\'"]?([^\'" >]+)', r.text)
    
    print("\n=== Forms ===")
    for f in forms[:10]:
        print(f"  {f}")
    
    print("\n=== Links with 'get' or 'download' ===")
    for l in links:
        if 'get' in l.lower() or 'download' in l.lower() or 'zip' in l.lower() or '.csv' in l.lower():
            print(f"  {l[:120]}")
    
    # Also search for the download mechanism
    print("\n=== Search for 'POST' or 'tk' or 'token' ===")
    for pattern in ['POST', 'action=', 'tk=', 'token=', 'download', 'file=']:
        matches = re.findall(f'{pattern}[^\'"]*', r.text, re.IGNORECASE)
        for m in matches[:5]:
            print(f"  {m[:100]}")
            
    # Save page source for inspection
    with open(r"E:\forex_data\cleaned\EURUSD\2000\_page_source.html", "w", encoding="utf-8") as f:
        f.write(r.text)
    print("\nPage source saved to _page_source.html")
    
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
