"""
MemoryMonitor — background thread logs process & system memory usage.
Logs to file + stdout every N seconds.
"""
import os
import time
import logging
import threading
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Background memory usage tracker.

    Usage:
        mon = MemoryMonitor(log_dir=r"E:\forex_data\logs", interval=30)
        mon.start()
        # ... run heavy work ...
        mon.stop()
    """

    def __init__(
        self,
        log_dir: str = r"E:\forex_data\logs",
        interval: int = 30,
        label: str = "memory",
    ):
        self.interval = interval
        self.label = label
        self._thread = None
        self._running = False

        # Setup log file
        self.log_dir = Path(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = self.log_dir / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        # Write CSV header
        with open(self.log_file, "w") as f:
            f.write("timestamp,process_mb,system_used_mb,system_total_mb,system_pct\n")

    def start(self):
        """Start background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("MemoryMonitor started -> %s (every %ds)", self.log_file, self.interval)

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("MemoryMonitor stopped. Log: %s", self.log_file)

    def _run(self):
        while self._running:
            try:
                proc_mb = 0
                sys_used = 0
                sys_total = 0
                sys_pct = 0

                if psutil:
                    proc = psutil.Process()
                    proc_mb = proc.memory_info().rss / (1024 * 1024)
                    mem = psutil.virtual_memory()
                    sys_used = mem.used / (1024 * 1024)
                    sys_total = mem.total / (1024 * 1024)
                    sys_pct = mem.percent

                ts = time.strftime("%Y-%m-%d %H:%M:%S")

                # Write to CSV
                with open(self.log_file, "a") as f:
                    f.write(f"{ts},{proc_mb:.1f},{sys_used:.1f},{sys_total:.1f},{sys_pct:.1f}\n")

                # Print to stdout
                logger.info(
                    "MEM | proc=%.0fMB | sys=%.0f/%.0fMB (%.1f%%)",
                    proc_mb, sys_used, sys_total, sys_pct,
                )

            except Exception:
                pass  # Never let monitor crash the main process

            time.sleep(self.interval)

    def log_once(self):
        """Log one snapshot (non-threaded, call inline)."""
        try:
            proc_mb = 0
            sys_used = 0
            sys_total = 0
            sys_pct = 0
            if psutil:
                proc = psutil.Process()
                proc_mb = proc.memory_info().rss / (1024 * 1024)
                mem = psutil.virtual_memory()
                sys_used = mem.used / (1024 * 1024)
                sys_total = mem.total / (1024 * 1024)
                sys_pct = mem.percent
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"{ts},{proc_mb:.1f},{sys_used:.1f},{sys_total:.1f},{sys_pct:.1f}\n")
            logger.info("MEM | proc=%.0fMB | sys=%.0f/%.0fMB (%.1f%%)",
                        proc_mb, sys_used, sys_total, sys_pct)
        except Exception:
            pass
