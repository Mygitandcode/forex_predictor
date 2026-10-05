"""Shared MySQL connection for API modules"""
import pymysql
from pymysql.cursors import DictCursor
import logging

log = logging.getLogger(__name__)

MYSQL_CONFIG = dict(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4",
    connect_timeout=3, read_timeout=10, write_timeout=10,
    cursorclass=DictCursor,
)


def get_conn():
    """Get a new MySQL connection"""
    return pymysql.connect(**MYSQL_CONFIG)


# Whitelist for dynamic table names (prevent SQL injection)
VALID_TFS = frozenset({"1m", "5m", "15m", "30m", "1h"})
VALID_SORT_FIELDS_RUNS = frozenset({"auc", "created_at", "live_accuracy"})
VALID_SORT_FIELDS_FV = frozenset({"created_at", "updated_at", "dimension", "feature_count"})
VALID_ORDERS = frozenset({"asc", "desc"})
