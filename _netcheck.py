import requests

session = requests.Session()
session.verify = False  # skip SSL for HTTP
requests.packages.urllib3.disable_warnings()

urls = [
    # Dukascopy HTTP
    "http://datafeed.dukascopy.com/datafeed/EURUSD/2025/01/02/12h_ticks.bi5",
    "http://www.dukascopy.com/datafeed/EURUSD/2025/01/02/12h_ticks.bi5",
    # Forex Tester data (alternative free source)
    "http://www.forextester.com/data",
]
for url in urls:
    try:
        r = session.get(url, timeout=10)
        print(f"OK [{r.status_code}]: {url[:80]}... -> {len(r.content)} bytes")
    except Exception as e:
        print(f"FAIL: {url[:60]}... -> {type(e).__name__[:20]}: {str(e)[:60]}")
