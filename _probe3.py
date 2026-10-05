"""Check the ASCII format form parameters."""
import requests, re

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

url = "http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2000"
r = session.get(url, timeout=15)

# Find the form
form_pattern = r'<form[^>]*action="/get\.php"[^>]*>(.*?)</form>'
forms = re.findall(form_pattern, r.text, re.DOTALL)
for i, f in enumerate(forms):
    print(f"=== Form {i} ===")
    inputs = re.findall(r'<input[^>]+>', f)
    for inp in inputs:
        name = re.search(r'name="([^"]+)"', inp)
        value = re.search(r'value="([^"]*)"', inp)
        if name:
            print(f"  {name.group(1)} = {value.group(1) if value else '(no value)'}")

# Also extract tk
tk_match = re.search(r'name="tk"\s+id="tk"\s+value="([^"]+)"', r.text)
print(f"\ntk token: {tk_match.group(1) if tk_match else 'NOT FOUND'}")

# Check the file listing mention
file_match = re.search(r'HISTDATA_COM[^\s"]*\.zip', r.text)
status_match = re.search(r'HISTDATA_COM[^\s"]*\.txt', r.text)
print(f"Expected ZIP name: {file_match.group(0) if file_match else 'NOT FOUND'}")
print(f"Status file name: {status_match.group(0) if status_match else 'NOT FOUND'}")
