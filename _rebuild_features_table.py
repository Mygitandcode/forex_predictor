"""Rebuild features_15m with all 212 feature columns + macd/rsi raw values"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pymysql
from pymysql.cursors import DictCursor
import pandas as pd, numpy as np
from features.feature_builder import FeatureBuilder

# Get feature names
builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)
df_fake = pd.DataFrame({
    'open': [1.1 + i*0.00001 for i in range(1000)],
    'high': [1.1001 + i*0.00001 for i in range(1000)],
    'low': [1.0999 + i*0.00001 for i in range(1000)],
    'close': [1.10005 + i*0.00001 for i in range(1000)],
    'tick_volume': [100]*1000,
})
df_fake.index = pd.date_range('2025-01-01', periods=1000, freq='1min')
df_fake['time'] = df_fake.index
_, feature_names = builder.build_many(df_fake)
print(f"Features: {len(feature_names)}")

# Build CREATE TABLE SQL
cols = [
    "timestamp DATETIME NOT NULL",
    "symbol VARCHAR(10) NOT NULL",
]
for i, name in enumerate(feature_names):
    safe_name = name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_')
    cols.append(f"`{safe_name}` FLOAT")
cols.append("target_15m TINYINT COMMENT '0=down 1=up -1=invalid'")
cols.append("PRIMARY KEY (timestamp, symbol)")
cols.append("INDEX idx_symbol (symbol)")

ddl = f"CREATE TABLE features_15m (\n  " + ",\n  ".join(cols) + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"

# Execute
conn = pymysql.connect(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")
cur = conn.cursor()
cur.execute("SET FOREIGN_KEY_CHECKS=0")
cur.execute("DROP TABLE IF EXISTS features_15m")
cur.execute("SET FOREIGN_KEY_CHECKS=1")
conn.commit()
cur.execute(ddl)
conn.commit()
cur.close()
conn.close()
print(f"features_15m rebuilt with {len(feature_names)} feature columns")
