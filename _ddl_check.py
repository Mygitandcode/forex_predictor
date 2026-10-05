"""Check existing table schemas before DDL"""
import urllib.request, json

API = "http://localhost:8000/api/mysql/query"

for tbl in ['feature_versions','model_runs','model_predictions','shap_contributions']:
    data = json.dumps({"sql": f"DESCRIBE {tbl}", "limit": 50}).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req)
    d = json.loads(r.read())
    print(f"=== {tbl} ===")
    if "detail" in d:
        print(f"  ERROR: {d['detail']}")
    elif "rows" not in d:
        print(f"  UNEXPECTED: {d}")
    else:
        for col in d["rows"]:
            print(f"  {col['Field']:25s} {col['Type']:25s} Null={col.get('Null',''):5s} Key={col.get('Key',''):5s} Default={col.get('Default','')}")
    print()
