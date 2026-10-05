"""Rebuild features_5m, features_30m, features_1h (same 212 cols as features_15m)"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import pymysql, pandas as pd, numpy as np
from features.feature_builder import FeatureBuilder

MYSQL = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")

builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)
df_fake = pd.DataFrame({'open':[1.1+i*0.00001 for i in range(1000)],'high':[1.1001+i*0.00001 for i in range(1000)],'low':[1.0999+i*0.00001 for i in range(1000)],'close':[1.10005+i*0.00001 for i in range(1000)],'tick_volume':[100]*1000})
df_fake.index = pd.date_range('2025-01-01', periods=1000, freq='1min')
df_fake['time'] = df_fake.index
_, feature_names = builder.build_many(df_fake)

safe_names = [n.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_') for n in feature_names]
cols = ["timestamp DATETIME NOT NULL", "symbol VARCHAR(10) NOT NULL"]
for sn in safe_names: cols.append(f"`{sn}` FLOAT")
cols.append("target_5m TINYINT")  # will rename per table
cols.append("PRIMARY KEY (timestamp, symbol)")
cols.append("INDEX idx_symbol (symbol)")

conn = pymysql.connect(**MYSQL)
cur = conn.cursor()
cur.execute("SET FOREIGN_KEY_CHECKS=0")

for table in ['features_5m', 'features_30m', 'features_1h']:
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    # Replace target column name
    tf_num = table.replace('features_','')
    cols_fixed = [c.replace('target_5m', f'target_{tf_num}') for c in cols]
    ddl = f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols_fixed) + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    cur.execute(ddl)
    conn.commit()
    print(f"  {table}: created with {len(feature_names)} cols")

cur.execute("SET FOREIGN_KEY_CHECKS=1")
cur.close(); conn.close()
print("All feature tables ready")
