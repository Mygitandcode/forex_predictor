r"""
hst_importer.py — Read MT4 .hst binary history files and convert to standardized DataFrame.

Uses mt4forexparser library under the hood.
Supports the "bo" suffix naming convention used by some brokers (e.g., EURUSDbo1.hst).

Usage:
    from data.hst_importer import HSTImporter
    importer = HSTImporter(
        hst_dir=r"C:\Users\...\history\FXOAK-Real01",
        output_dir=r"D:\dapanAllThings\datas"
    )
    df = importer.import_symbol("EURUSD")

Tier 2 in data acquisition pipeline:
    Dukascopy (long-term) -> MT4 HST (recent) -> ZeroMQ (real-time, future)
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd

from mt4forexparser.MT4ForexParser import MT4ParseToPD

logger = logging.getLogger(__name__)

# Suppress mt4forexparser's internal logger spam (prints 5 INFO lines per file read)
logging.getLogger("MT4ForexParser").setLevel(logging.WARNING)

# MT4 timeframe codes -> minute mapping
TIMEFRAME_MAP = {
    "1": "M1",
    "5": "M5",
    "15": "M15",
    "30": "M30",
    "60": "H1",
    "240": "H4",
    "1440": "D1",
    "10080": "W1",
    "43200": "MN1",
}

# HST filename patterns to try per symbol
HST_PATTERNS = [
    "{symbol}bo1.hst",    # Broker-specific: EURUSDbo1.hst
    "{symbol}1.hst",      # Standard MT4: EURUSD1.hst
    "{symbol}_1.hst",
]


class HSTImporter:
    """
    Read MT4 .hst binary history files.

    Scans the MT4 history directory for HST files per symbol,
    reads them via mt4forexparser, and saves standardized CSV.
    
    IDEMPOTENT: Uses file hash to detect changes. Only re-imports
    when the HST file has been updated.
    """

    def __init__(self, hst_dir: str, output_dir: str):
        """
        Args:
            hst_dir: MT4 history directory containing .hst files
                     e.g. C:\\Users\\...\\MetaQuotes\\Terminal\\...\\history\\FXOAK-Real01
            output_dir: Where to save standardized CSV output
                        e.g. D:\\dapanAllThings\\datas
        """
        self.hst_dir = Path(hst_dir)
        self.output_dir = Path(output_dir)
        self.hst_output_dir = self.output_dir / "mt4_hst"
        self.hash_file = self.output_dir / "manifest_hst_hashes.json"

        self._ensure_dirs()
        self._hashes = self._load_hashes()

    def list_symbols(self) -> List[str]:
        """Scan HST directory and return list of unique symbols with M1 data."""
        symbols = set()
        if not self.hst_dir.exists():
            logger.warning("HST directory not found: %s", self.hst_dir)
            return []

        for f in self.hst_dir.glob("*.hst"):
            name = f.name
            # Extract symbol from filename like "EURUSDbo1.hst" or "EURUSD1.hst"
            for suffix in ["bo1.hst", "1.hst", "_1.hst"]:
                if name.endswith(suffix):
                    symbol = name[:-len(suffix)]
                    symbols.add(symbol)
                    break
            else:
                # Fallback: try to extract known symbol patterns
                for known in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
                              "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
                              "AUDJPY", "NZDJPY"]:
                    if name.startswith(known) and "bo1" in name:
                        symbols.add(known)
                        break

        return sorted(symbols)

    def import_symbol(
        self,
        symbol: str,
        force: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Import one symbol's M1 data from HST files.

        Args:
            symbol: e.g. "EURUSD"
            force: If True, re-import even if hash hasn't changed

        Returns:
            DataFrame with standardized OHLCV columns, or None if no data
        """
        # ---- Find HST file ----
        hst_path = self._find_hst_file(symbol)
        if hst_path is None:
            logger.warning("[%s] No HST file found in %s", symbol, self.hst_dir)
            return None

        # ---- Idempotency check via file hash ----
        file_hash = self._file_hash(hst_path)
        if not force and self._hashes.get(symbol) == file_hash:
            logger.info("[%s] HST file unchanged (hash: %s). Skipping.", symbol, file_hash[:8])
            # Still try to load existing output
            return self._load_existing_csv(symbol)

        logger.info("[%s] Reading HST: %s", symbol, hst_path)

        # ---- Parse HST file ----
        try:
            df = MT4ParseToPD(str(hst_path))
        except Exception as e:
            logger.error("[%s] Failed to parse HST file: %s", symbol, e)
            return None

        if df is None or df.empty:
            logger.warning("[%s] HST file parsed but returned empty DataFrame.", symbol)
            return None

        logger.info("[%s] Parsed %d rows from HST.", symbol, len(df))

        # ---- Standardize columns to match csv_importer output ----
        df = self._standardize(df, symbol)

        # ---- Save standardized CSV ----
        self._save_csv(df, symbol)

        # ---- Update hash manifest ----
        self._hashes[symbol] = file_hash
        self._save_hashes()

        return df

    def import_all(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Import all available symbols (or specified subset).
        
        Returns:
            {symbol: DataFrame} dict
        """
        if symbols is None:
            symbols = self.list_symbols()

        results = {}
        for symbol in symbols:
            try:
                df = self.import_symbol(symbol, force=force)
                results[symbol] = df
            except Exception as e:
                logger.error("[%s] Import failed: %s", symbol, e, exc_info=True)
                results[symbol] = None

        imported = sum(1 for v in results.values() if v is not None)
        logger.info("HST import complete: %d/%d symbols imported.", imported, len(symbols))
        return results

    # ---- Internal methods ----

    def _ensure_dirs(self):
        """Recursively create required directories."""
        try:
            os.makedirs(self.hst_output_dir, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create HST output dir %s: %s", self.hst_output_dir, e)

    def _find_hst_file(self, symbol: str) -> Optional[Path]:
        """Find the M1 HST file for a symbol."""
        if not self.hst_dir.exists():
            return None

        for pattern in HST_PATTERNS:
            fname = pattern.format(symbol=symbol)
            fpath = self.hst_dir / fname
            if fpath.exists():
                return fpath

        return None

    def _file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file for change detection."""
        try:
            sha = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception as e:
            logger.warning("Failed to hash file %s: %s", path, e)
            return ""

    def _load_hashes(self) -> Dict[str, str]:
        """Load HST file hash manifest."""
        try:
            if self.hash_file.exists():
                import json
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug("No existing HST hash manifest: %s", e)
        return {}

    def _save_hashes(self):
        """Save HST file hash manifest."""
        try:
            import json
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(self._hashes, f, indent=2)
        except Exception as e:
            logger.error("Failed to save HST hash manifest: %s", e)

    def _load_existing_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try to load previously-imported CSV for a symbol."""
        csv_path = self.hst_output_dir / f"{symbol}_M1.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime')
                logger.debug("[%s] Loaded existing CSV: %d rows", symbol, len(df))
                return df
            except Exception as e:
                logger.warning("[%s] Failed to load existing CSV: %s", symbol, e)
        return None

    def _standardize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardize mt4forexparser output to match csv_importer format.

        Expected input columns: date, time, open, high, low, close, volume
        Output: datetime index, columns: open, high, low, close, tick_volume
        """
        # Build datetime index from date + time columns
        if 'date' in df.columns and 'time' in df.columns:
            try:
                datetimes = pd.to_datetime(
                    df['date'].astype(str) + ' ' + df['time'].astype(str),
                    format='%Y.%m.%d %H:%M'
                )
                df['datetime'] = datetimes
                df = df.drop(columns=['date', 'time'])
            except Exception:
                # Keep date/time if parsing fails
                pass

        # Set datetime as index if available
        if 'datetime' in df.columns:
            df = df.set_index('datetime')

        # Rename volume -> tick_volume for consistency
        if 'volume' in df.columns:
            df = df.rename(columns={'volume': 'tick_volume'})

        # Ensure required columns are numeric
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic cleaning
        df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')
        if 'tick_volume' in df.columns:
            df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce').fillna(0).astype(int)

        # Remove invalid rows
        if all(c in df.columns for c in ['high', 'low']):
            df = df[df['high'] >= df['low']]

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _save_csv(self, df: pd.DataFrame, symbol: str):
        """Save standardized DataFrame to CSV."""
        if df.empty:
            return

        csv_path = self.hst_output_dir / f"{symbol}_M1.csv"
        try:
            df.to_csv(csv_path)
            logger.info("[%s] Saved standardized CSV: %s (%d rows)",
                        symbol, csv_path, len(df))
        except Exception as e:
            logger.error("[%s] Failed to save CSV: %s", symbol, e)


# ---- Quick test ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    importer = HSTImporter(
        hst_dir=r"C:\Users\admin\AppData\Roaming\MetaQuotes\Terminal\7F9720261EB20B849E02C5EEE9135BF9\history\FXOAK-Real01",
        output_dir=r"E:\forex_data"
    )
    symbols = importer.list_symbols()
    print(f"Found {len(symbols)} symbols: {symbols}")
    if symbols:
        importer.import_symbol(symbols[0])
