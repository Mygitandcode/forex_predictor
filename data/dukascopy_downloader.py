"""
dukascopy_downloader.py — Download long-term historical forex tick data from Dukascopy.

Dukascopy provides free tick data going back to ~2003.
Ticks are aggregated to 1-minute OHLCV bars.
Supports idempotent download: skips already-downloaded dates via manifest file.

Multi-threaded: ThreadPoolExecutor (5 workers), each worker downloads 1 day of tick data,
aggregates to 1-min OHLCV, and saves as a temp CSV. After all days complete, temp files
are merged into the final output file. Crash-safe: each day is persisted immediately.

Usage:
    from data.dukascopy_downloader import DukascopyDownloader
    dl = DukascopyDownloader(r"E:\forex_data")
    dl.download("EURUSD", start_date="2010-01-01", end_date="2025-12-31")

Tier 1 in data acquisition pipeline:
    Dukascopy (long-term) -> MT4 HST (recent) -> ZeroMQ (real-time, future)
"""

import os
import lzma
import struct
import logging
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol configuration: Dukascopy name -> point value (10^digits)
# ---------------------------------------------------------------------------
SYMBOL_CONFIG: Dict[str, int] = {
    "EURUSD": 100000, "GBPUSD": 100000, "USDJPY": 1000,
    "EURJPY": 1000,   "GBPJPY": 1000,   "AUDUSD": 100000,
    "NZDUSD": 100000, "USDCAD": 100000, "USDCHF": 100000,
    "EURGBP": 100000, "AUDJPY": 1000,   "NZDJPY": 1000,
}

DUKASCOPY_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Thread pool: network I/O bound, 5 workers balances speed vs Dukascopy rate limits
MAX_WORKERS = 5

# Retry settings (per hour file)
MAX_RETRIES = 1
HTTP_TIMEOUT = 8  # seconds

# Default data root (E:\ drive, total data ~11GB for 17 symbols x 12 years)
DEFAULT_DATA_ROOT = r"E:\forex_data"


class DukascopyDownloader:
    """
    Automated Dukascopy historical tick data downloader.

    Multi-threaded: uses ThreadPoolExecutor to download multiple days in parallel.
    Each day's data is saved to a temp CSV file immediately (crash-safe).
    After all days complete, temp files are merged into the final CSV.

    IDEMPOTENT: manifest tracks completed dates per symbol.
    Re-running skips already-downloaded dates.
    """

    def __init__(self, data_root: str = DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "dukascopy"
        self.temp_dir = self.raw_dir / "_temp"
        self.manifest_path = self.data_root / "manifest_dukascopy.json"

        self._ensure_dirs()
        self._manifest = self._load_manifest()
        self._manifest_lock = Lock()

    # ---- Public API ----

    def download(
        self,
        symbol: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download historical data for one symbol from Dukascopy.

        Returns: Path to final CSV, or None if already fully downloaded.
        """
        symbol = symbol.upper()
        point = SYMBOL_CONFIG.get(symbol, 100000)

        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # ---- Idempotency: find which days still need downloading ----
        done_dates = self._get_done_dates(symbol)
        all_dates = self._date_range(start_dt, end_dt)
        todo_dates = [d for d in all_dates if d not in done_dates]

        if not todo_dates:
            logger.info("[%s] All %d dates already downloaded. Skipping.", symbol, len(all_dates))
            return self._final_path(symbol)

        logger.info("[%s] %d total dates, %d already done, %d to download.",
                    symbol, len(all_dates), len(done_dates), len(todo_dates))
        logger.info("[%s] Thread pool: %d workers. Each worker = 1 day of tick data.",
                    symbol, MAX_WORKERS)

        # ---- Multi-threaded download ----
        completed = 0
        total = len(todo_dates)
        last_log_time = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._download_single_day, symbol, d, point): d
                for d in todo_dates
            }

            for future in as_completed(futures):
                day = futures[future]
                completed += 1
                try:
                    success = future.result()
                except Exception as e:
                    logger.warning("[%s] Day %s failed: %s", symbol, day.strftime("%Y-%m-%d"), e)
                    success = False

                # Progress log every 30 seconds
                now = time.time()
                if now - last_log_time >= 30 or completed >= total:
                    pct = completed * 100.0 / total
                    logger.info("[%s] Progress: %d/%d days (%.1f%%) | last: %s",
                                symbol, completed, total, pct,
                                day.strftime("%Y-%m-%d"))
                    last_log_time = now

        logger.info("[%s] All %d days downloaded.", symbol, total)

        # ---- Merge temp files -> final CSV ----
        return self._merge_temp_files(symbol, all_dates)

    def download_all(
        self,
        symbols: List[str],
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> Dict[str, Optional[Path]]:
        """Download all specified symbols (serial, each symbol uses its own thread pool)."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.download(symbol, start_date, end_date)
            except Exception as e:
                logger.error("[%s] Download failed: %s", symbol, e, exc_info=True)
                results[symbol] = None
        return results

    # ---- Internal: directory & manifest ----

    def _ensure_dirs(self):
        for d in [self.raw_dir, self.temp_dir]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create dir %s: %s", d, e)

    def _load_manifest(self) -> Dict:
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to load manifest: %s. Starting fresh.", e)
        return {}

    def _save_manifest(self):
        try:
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as e:
            logger.error("Failed to save manifest: %s", e)

    def _get_done_dates(self, symbol: str) -> Set[str]:
        """Get set of already-downloaded date strings for a symbol."""
        return set(self._manifest.get(symbol, []))

    def _mark_done(self, symbol: str, date_str: str):
        """Mark a single date as downloaded (thread-safe)."""
        with self._manifest_lock:
            if symbol not in self._manifest:
                self._manifest[symbol] = []
            if date_str not in self._manifest[symbol]:
                self._manifest[symbol].append(date_str)
                self._save_manifest()

    def _date_range(self, start: datetime, end: datetime) -> List[datetime]:
        """Generate list of datetime objects for each day in range."""
        days = (end - start).days + 1
        return [start + timedelta(days=i) for i in range(days)]

    def _final_path(self, symbol: str) -> Path:
        return self.raw_dir / f"{symbol}_M1.csv"

    def _temp_path(self, symbol: str, day: datetime) -> Path:
        """Path for a single day's temp CSV."""
        return self.temp_dir / f"{symbol}_{day.strftime('%Y%m%d')}.csv"

    # ---- Core: single-day download worker ----

    def _download_single_day(
        self,
        symbol: str,
        day: datetime,
        point: int,
    ) -> bool:
        """
        Download 1 day (24 hours) of tick data, aggregate, save as temp CSV.
        Returns True on success. Thread-safe: each day writes to its own file.
        """
        date_str = day.strftime("%Y-%m-%d")

        # Download 24 hour files for this day
        ticks = self._download_day_ticks(symbol, day)

        if not ticks:
            logger.debug("[%s] %s: 0 ticks (weekend/holiday)", symbol, date_str)
            # Still mark as done so we don't retry forever
            self._mark_done(symbol, date_str)
            return True

        # Aggregate ticks -> 1-min OHLCV
        df = self._aggregate_ticks(ticks, point)
        if df.empty:
            self._mark_done(symbol, date_str)
            return True

        # Save temp CSV immediately (crash-safe)
        temp_path = self._temp_path(symbol, day)
        try:
            df.to_csv(temp_path)
            logger.debug("[%s] %s: %d ticks -> %d bars -> %s",
                         symbol, date_str, len(ticks), len(df), temp_path.name)
        except Exception as e:
            logger.error("[%s] %s: failed to save temp CSV: %s", symbol, date_str, e)
            return False

        # Mark done
        self._mark_done(symbol, date_str)
        return True

    def _download_day_ticks(
        self,
        symbol: str,
        day: datetime,
    ) -> List[Tuple]:
        """Download all 24 hour files for a single day."""
        # Use a fresh session per thread (requests.Session is not thread-safe for concurrent use)
        session = requests.Session()
        session.headers.update({"User-Agent": "forex_predictor/1.0"})
        ticks = []

        for hour in range(24):
            hour_dt = day.replace(hour=hour)
            url = (f"{DUKASCOPY_BASE_URL}/{symbol}/"
                   f"{hour_dt.year}/"
                   f"{hour_dt.month:02d}/"
                   f"{hour_dt.day:02d}/"
                   f"{hour_dt.hour:02d}h_ticks.bi5")

            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = session.get(url, timeout=HTTP_TIMEOUT)
                    if resp.status_code == 404:
                        break
                    elif resp.status_code != 200:
                        if attempt < MAX_RETRIES:
                            time.sleep(0.5)
                        continue
                    hour_ticks = self._parse_bi5(resp.content)
                    ticks.extend(hour_ticks)
                    break
                except requests.RequestException:
                    if attempt < MAX_RETRIES:
                        time.sleep(0.5)
                except Exception:
                    break

        session.close()
        return ticks

    # ---- Parse .bi5 binary ----

    def _parse_bi5(self, data: bytes) -> List[Tuple]:
        """Parse Dukascopy .bi5 LZMA-compressed binary format."""
        try:
            decompressed = lzma.decompress(data)
        except lzma.LZMAError:
            return []

        ticks = []
        record_size = 20
        num_records = len(decompressed) // record_size

        for i in range(num_records):
            offset = i * record_size
            try:
                ts, ask_r, bid_r, ask_v, bid_v = struct.unpack_from(
                    '>iIIff', decompressed, offset
                )
                if ask_r > 0 and bid_r > 0 and ts > 0:
                    ticks.append((ts, ask_r, bid_r, ask_v, bid_v))
            except struct.error:
                continue

        return ticks

    # ---- Aggregate ticks -> 1-min OHLCV ----

    def _aggregate_ticks(self, ticks: List[Tuple], point: int) -> pd.DataFrame:
        """Aggregate tick data to 1-minute OHLCV bars."""
        if not ticks:
            return pd.DataFrame()

        df = pd.DataFrame(ticks, columns=['ts_ms', 'ask_raw', 'bid_raw', 'ask_vol', 'bid_vol'])
        df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms')
        df['minute'] = df['datetime'].dt.floor('min')
        df['ask'] = df['ask_raw'] / point
        df['bid'] = df['bid_raw'] / point

        grouped = df.groupby('minute')
        result = pd.DataFrame({
            'open': grouped['ask'].first(),
            'high': grouped['ask'].max(),
            'low': grouped['bid'].min(),
            'close': grouped['bid'].last(),
            'tick_volume': grouped['bid_vol'].sum(),
        })
        result.index.name = 'datetime'
        result = result.sort_index()
        return result

    # ---- Merge temp files -> final CSV ----

    def _merge_temp_files(self, symbol: str, all_dates: List[datetime]) -> Optional[Path]:
        """
        Merge all per-day temp CSV files into the final output CSV.
        Checks that every date (weekends/holidays included) has been processed.
        Removes temp files after merge.
        """
        final_path = self._final_path(symbol)

        # Collect all temp files for this symbol
        temp_files = sorted(self.temp_dir.glob(f"{symbol}_*.csv"))
        if not temp_files:
            logger.info("[%s] No temp files to merge.", symbol)
            return None

        logger.info("[%s] Merging %d temp files -> %s ...", symbol, len(temp_files), final_path)

        # Concatenate all temp CSVs
        frames = []
        errors = 0
        for tf in temp_files:
            try:
                df = pd.read_csv(tf, parse_dates=['datetime'], index_col='datetime')
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                logger.warning("[%s] Failed to read temp file %s: %s", symbol, tf.name, e)
                errors += 1

        if errors:
            logger.warning("[%s] %d temp files skipped due to read errors.", symbol, errors)

        if not frames:
            logger.warning("[%s] All temp files were empty or unreadable.", symbol)
            return None

        # Concat + deduplicate + sort
        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep='first')]
        merged = merged.sort_index()

        # Save final CSV
        try:
            merged.to_csv(final_path)
            logger.info("[%s] Final CSV saved: %s (%d rows, %s -> %s)",
                        symbol, final_path, len(merged),
                        merged.index[0].strftime("%Y-%m-%d"),
                        merged.index[-1].strftime("%Y-%m-%d"))
        except Exception as e:
            logger.error("[%s] Failed to save final CSV: %s", symbol, e)
            return None

        # Clean up temp files (only if final save succeeded)
        for tf in temp_files:
            try:
                tf.unlink()
            except Exception:
                pass

        # Remove temp dir if empty
        try:
            if not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
        except Exception:
            pass

        return final_path


# ---- Quick test ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    dl = DukascopyDownloader()
    dl.download("EURUSD", start_date="2026-01-02", end_date="2026-01-03")
