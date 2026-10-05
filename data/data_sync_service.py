"""
data_sync_service.py — Main data acquisition orchestrator.

Acts as a daemon-like service:
  1. Dukascopy (Tier 1): Download long-term historical data (2003+)
  2. MT4 HST   (Tier 2): Import recent broker data from .hst files
  3. ZeroMQ    (Tier 3): Real-time streaming (future)

IDEMPOTENT: Manifest files prevent re-downloading already-handled data.
RESILIENT: Every operation is wrapped in try/except. Missing dirs auto-created.
            Empty folders or missing data gracefully skipped.

Data flow:
  Dukascopy --> datas/dukascopy/{symbol}_M1.csv   (long-term)
  MT4 HST   --> datas/mt4_hst/{symbol}_M1.csv     (recent)
  Merger    --> datas/cleaned/{symbol}_M1.parquet (combined)

Usage:
  Start service:   python -m data.data_sync_service
  Once & exit:     python -m data.data_sync_service --once

From PyCharm: just run this file directly.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# Ensure project root is on path (for PyCharm / direct execution)
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.dukascopy_downloader import DukascopyDownloader
from data.hst_importer import HSTImporter

logger = logging.getLogger(__name__)

# ---- Configuration ----

# All currency pairs to track
# Source: config/symbols.yaml primary pairs + extras from HST scan
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "NZDJPY",
]

# Data directories (~11GB for 17 symbols x 12 years -> E:\ drive)
DATA_ROOT = r"E:\forex_data"

# MT4 HST directory
HST_DIR = r"C:\Users\admin\AppData\Roaming\MetaQuotes\Terminal\7F9720261EB20B849E02C5EEE9135BF9\history\FXOAK-Real01"

# How often to check for new data (seconds) when running as daemon
POLL_INTERVAL = 300  # 5 minutes

# How far back to download from Dukascopy
DUKASCOPY_START = "2014-01-01"  # ~12 years of data (2026 - 2014)


class DataSyncService:
    """
    Main data synchronization service.

    Orchestrates Dukascopy download + MT4 HST import + data merging.
    Designed to run continuously, polling for new data periodically.
    """

    def __init__(
        self,
        data_root: str = DATA_ROOT,
        hst_dir: str = HST_DIR,
        symbols: Optional[List[str]] = None,
        poll_interval: int = POLL_INTERVAL,
    ):
        self.data_root = Path(data_root)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.hst_dir = hst_dir
        self.poll_interval = poll_interval

        # Subdirs
        self.cleaned_dir = self.data_root / "cleaned"

        # Components
        self.dukascopy = DukascopyDownloader(data_root)
        self.hst = HSTImporter(hst_dir, data_root)

        # Ensure all directories exist
        self._ensure_all_dirs()

        self._running = True

    # ---- Public API ----

    def sync_once(self) -> Dict[str, dict]:
        """
        Run one full synchronization cycle.

        1. Dukascopy: download long-term history
        2. MT4 HST: import recent broker data
        3. Merge: combine both sources, deduplicate, save cleaned file

        Returns:
            {symbol: {"dukascopy": rows, "hst": rows, "merged": rows}}
        """
        results = {}

        # ---- Step 1: Dukascopy download (long-term) ----
        logger.info("=" * 50)
        logger.info("Phase 1/3: Dukascopy download (long-term history)")
        logger.info("=" * 50)
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        for symbol in self.symbols:
            try:
                logger.info("--- [%s] Dukascopy download ---", symbol)
                duka_path = self.dukascopy.download(
                    symbol,
                    start_date=DUKASCOPY_START,
                    end_date=end_date,
                )
                results.setdefault(symbol, {})['dukascopy'] = duka_path
            except Exception as e:
                logger.error("[%s] Dukascopy phase failed: %s", symbol, e)
                results.setdefault(symbol, {})['dukascopy'] = f"ERROR: {e}"

        # ---- Step 2: MT4 HST import (recent data) ----
        logger.info("=" * 50)
        logger.info("Phase 2/3: MT4 HST import (recent broker data)")
        logger.info("=" * 50)

        # Discover available HST symbols (may differ from full symbol list)
        hst_symbols = self.hst.list_symbols()
        if hst_symbols:
            logger.info("HST symbols found: %s", hst_symbols)
        else:
            logger.warning("No HST files found in %s. Skipping HST phase.", self.hst_dir)

        for symbol in hst_symbols:
            try:
                logger.info("--- [%s] HST import ---", symbol)
                df = self.hst.import_symbol(symbol)
                results.setdefault(symbol, {})['hst'] = len(df) if df is not None else 0
            except Exception as e:
                logger.error("[%s] HST phase failed: %s", symbol, e)
                results.setdefault(symbol, {})['hst'] = f"ERROR: {e}"

        # ---- Step 3: Merge Dukascopy + HST data ----
        logger.info("=" * 50)
        logger.info("Phase 3/3: Merging Dukascopy + HST data")
        logger.info("=" * 50)

        for symbol in self.symbols:
            try:
                merged_rows = self._merge_symbol(symbol)
                results.setdefault(symbol, {})['merged'] = merged_rows
            except Exception as e:
                logger.error("[%s] Merge phase failed: %s", symbol, e)
                results.setdefault(symbol, {})['merged'] = f"ERROR: {e}"

        # ---- Summary ----
        logger.info("=" * 50)
        logger.info("Sync cycle complete. Summary:")
        for symbol, info in sorted(results.items()):
            parts = []
            for k, v in info.items():
                if isinstance(v, int):
                    parts.append(f"{k}={v} rows")
                elif v is not None:
                    parts.append(f"{k}=OK")
                else:
                    parts.append(f"{k}=None")
            logger.info("  %-10s | %s", symbol, " | ".join(parts))
        logger.info("=" * 50)

        return results

    def run_daemon(self):
        """
        Run as a persistent service.

        Executes sync_once() on startup, then polls HST_DIR periodically
        for new data and merges any changes.
        """
        logger.info("DataSyncService daemon starting...")
        logger.info("Data root: %s", self.data_root)
        logger.info("HST dir:  %s", self.hst_dir)
        logger.info("Symbols:  %s", self.symbols)
        logger.info("Poll interval: %d seconds", self.poll_interval)

        # First run: full sync
        self.sync_once()

        logger.info("Entering polling loop (Ctrl+C to stop)...")
        iteration = 0

        while self._running:
            try:
                iteration += 1
                time.sleep(self.poll_interval)

                logger.info("Poll iteration %d: checking for new HST data...", iteration)

                # Only poll HST (Dukascopy is fixed-range, no need to re-download)
                hst_symbols = self.hst.list_symbols()
                if not hst_symbols:
                    logger.debug("No HST symbols found, HST dir may be empty.")
                    continue

                for symbol in hst_symbols:
                    try:
                        df = self.hst.import_symbol(symbol)
                        if df is not None:
                            # Merge the newly imported HST data
                            self._merge_symbol(symbol)
                    except Exception as e:
                        logger.error("[%s] Poll import failed: %s", symbol, e)

            except KeyboardInterrupt:
                logger.info("Received interrupt. Shutting down...")
                break
            except Exception as e:
                logger.error("Poll iteration %d failed: %s", iteration, e, exc_info=True)
                # Continue running despite errors

        logger.info("DataSyncService stopped.")

    def stop(self):
        """Graceful shutdown."""
        self._running = False

    # ---- Internal ----

    def _ensure_all_dirs(self):
        """Recursively create all required directories."""
        dirs_to_create = [
            self.data_root,
            self.cleaned_dir,
        ]
        for d in dirs_to_create:
            try:
                os.makedirs(d, exist_ok=True)
                logger.debug("Directory ready: %s", d)
            except Exception as e:
                logger.error("Failed to create directory %s: %s", d, e)

    def _merge_symbol(self, symbol: str) -> int:
        """
        Merge Dukascopy (long-term) + HST (recent) data for one symbol.

        Strategy:
          - Load both sources
          - Concatenate
          - Sort by time
          - Remove duplicates (keep HST for overlaps, as it's broker-confirmed)
          - Basic cleaning
          - Save as Parquet to cleaned/

        Returns:
            Number of rows in merged dataset
        """
        dfs = []

        # Try loading Dukascopy data
        duka_path = self.dukascopy.raw_dir / f"{symbol}_M1.csv"
        if duka_path.exists():
            try:
                duka_df = pd.read_csv(duka_path, parse_dates=['datetime'], index_col='datetime')
                if len(duka_df) > 0:
                    dfs.append(('dukascopy', duka_df))
            except Exception as e:
                logger.warning("[%s] Failed to load Dukascopy CSV: %s", symbol, e)

        # Try loading HST data
        hst_path = self.hst.hst_output_dir / f"{symbol}_M1.csv"
        if hst_path.exists():
            try:
                hst_df = pd.read_csv(hst_path, parse_dates=['datetime'], index_col='datetime')
                if len(hst_df) > 0:
                    dfs.append(('hst', hst_df))
            except Exception as e:
                logger.warning("[%s] Failed to load HST CSV: %s", symbol, e)

        if not dfs:
            logger.debug("[%s] No data to merge.", symbol)
            return 0

        # Merge: HST takes priority where they overlap (broker data is more reliable)
        if len(dfs) == 1:
            merged = dfs[0][1].copy()
        else:
            # Concat, keep last (HST) where duplicates
            combined = pd.concat([df[1] for df in dfs])
            merged = combined[~combined.index.duplicated(keep='last')]
            merged = merged.sort_index()

        # Basic cleaning
        merged = merged.dropna(subset=['open', 'high', 'low', 'close'], how='all')
        if all(c in merged.columns for c in ['high', 'low']):
            merged = merged[merged['high'] >= merged['low']]

        # Save as Parquet (efficient compressed format)
        output_path = self.cleaned_dir / f"{symbol}_M1.parquet"
        try:
            merged.to_parquet(output_path, compression='snappy')
        except Exception as e:
            logger.error("[%s] Failed to save merged Parquet: %s", symbol, e)
            return len(merged)

        # Show date range
        if len(merged) > 0:
            date_range = f"{merged.index[0]} -> {merged.index[-1]}"
        else:
            date_range = "empty"

        logger.info("[%s] Merged: %d rows (%s) -> %s",
                    symbol, len(merged), date_range, output_path)

        return len(merged)


# ---- CLI Entry Point ----

def main():
    parser = argparse.ArgumentParser(
        description="DataSyncService - Multi-tier forex data acquisition"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one sync cycle and exit (default: run as daemon)"
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Specific symbols to sync, e.g. --symbols EURUSD GBPUSD"
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL,
        help=f"Poll interval in seconds for daemon mode (default: {POLL_INTERVAL})"
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help=f"Data root directory (default: {DATA_ROOT})"
    )
    parser.add_argument(
        "--hst-dir", default=HST_DIR,
        help="MT4 history directory with .hst files"
    )
    args = parser.parse_args()

    # Configure logging: simple English, no Unicode issues
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    poll_interval = args.interval

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    service = DataSyncService(
        data_root=args.data_root,
        hst_dir=args.hst_dir,
        symbols=symbols,
        poll_interval=poll_interval,
    )

    if args.once:
        logger.info("Running single sync cycle...")
        service.sync_once()
        logger.info("Sync complete. Exiting.")
    else:
        service.run_daemon()


if __name__ == "__main__":
    main()
