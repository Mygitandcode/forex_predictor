"""Verify histdata EURUSD 2000 CSV quality."""
import pandas as pd

csv_path = r"E:\forex_data\cleaned\EURUSD\2000\DAT_ASCII_EURUSD_M1_2000.csv"
# Format: YYYYMMDD HHMMSS;open;high;low;close;volume (semicolon-separated)
df = pd.read_csv(csv_path, sep=';', header=None,
                 names=['date', 'open', 'high', 'low', 'close', 'volume'])
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nLast 3 rows:\n{df.tail(3)}")

# Parse datetime
df['first_dt'] = pd.to_datetime(df['date'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
print(f"\nDate range: {df['first_dt'].min()} -> {df['first_dt'].max()}")

# Check OHLC sanity
print(f"\nPrice stats:")
for col in ['open', 'high', 'low', 'close']:
    print(f"  {col}: min={df[col].min():.5f}  max={df[col].max():.5f}  mean={df[col].mean():.5f}")

# Check for zeros
for col in ['open', 'high', 'low', 'close']:
    zeros = (df[col] == 0).sum()
    if zeros:
        print(f"  WARNING: {col} has {zeros} zeros!")
        
print(f"\nVolume > 0: {(df['volume'] > 0).sum()} / {len(df)}")

# Check time gaps (should be ~1 minute in theory but this is tick data)
df['datetime'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d %H%M%S', errors='coerce')
df = df.sort_values('datetime')
gaps = df['datetime'].diff().dropna()
print(f"\nTime gaps (seconds):")
print(f"  Mean: {gaps.dt.total_seconds().mean():.0f}s")
print(f"  Median: {gaps.dt.total_seconds().median():.0f}s")
print(f"  Max: {gaps.dt.total_seconds().max():.0f}s")

# Show distribution of gap sizes
large_gaps = gaps[gaps > pd.Timedelta(minutes=5)]
print(f"  Gaps > 5min: {len(large_gaps)}")
if len(large_gaps) > 0:
    print(f"  Largest 5 gaps:")
    for idx in large_gaps.sort_values(ascending=False).head(5).index:
        print(f"    {idx}: {large_gaps[idx]}")
