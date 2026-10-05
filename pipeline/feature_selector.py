"""
pipeline/feature_selector.py — 从 feature_importance_report.md 提取有效特征

提供:
  - get_effective_features(top_n=None) → 317 个 gain>0 特征列表
  - get_top_features(n) → Top-N 特征
  - fallback 模式: 如果 report 不可用，读 features/feature_builder.py 列首部

设计蓝图: dapanPlan/guides/mq-pipeline-blueprint.md §四
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import List, Optional

# 确保项目根在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# 默认路径 (相对项目根)
_REPORT_PATH = "docs/feature_importance_report.md"


def get_effective_features(top_n: Optional[int] = None) -> List[str]:
    """从 feature_importance_report.md 提取有效特征名。

    解析 §六 中的完整排名表，格式:
    | rank | fXXX | gain | name |

    Args:
        top_n: 取前 N 个，None = 全部 317 个

    Returns:
        特征名列表 (按 gain 降序)
    """
    report = _find_report()
    if report is None:
        logger.warning("feature_importance_report.md 未找到，返回空列表")
        return []

    features = _parse_report(report)
    if not features:
        logger.warning("未能从 report 解析到有效特征")
        return []

    logger.info(f"从 report 解析到 {len(features)} 个有效特征")

    if top_n and top_n < len(features):
        return features[:top_n]

    return features


def get_top_features(n: int) -> List[str]:
    """获取 Top-N 特征"""
    return get_effective_features(top_n=n)


def _find_report() -> Optional[str]:
    """定位 feature_importance_report.md"""
    # 从当前文件向上找到项目根，再拼接 docs/
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)  # pipeline/ → project root
    path = os.path.join(root, _REPORT_PATH)
    if os.path.exists(path):
        return path

    # fallback: 从 CWD 找
    path = os.path.join(os.getcwd(), _REPORT_PATH)
    if os.path.exists(path):
        return path

    return None


def _parse_report(path: str) -> List[str]:
    """从 Markdown 报告 §六 中提取特征排名表"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # §六 位于 "## 六、完整特征排名" 之后
    # 表格格式: | 1 | f208 | 1924.0 | 1m_close_ma60_ratio |
    section_six = _extract_section(content, "六、完整特征排名")
    if not section_six:
        logger.warning("报告中未找到 §六、完整特征排名")
        return []

    features = []
    in_table = False
    for line in section_six.split("\n"):
        line = line.strip()

        # 检测表格行: | rank | fXXX | gain | name |
        if line.startswith("|") and "|" in line[1:]:
            parts = [p.strip() for p in line.split("|")]
            # 跳过表头: | Rank | Feature | Gain | 含义...
            if len(parts) >= 5 and parts[1].isdigit():
                try:
                    rank = int(parts[1])
                    # | Rank | Feature | Gain |
                    # parts = ['', rank, ' `feature_name` ', ' gain ']
                    # 特征名位于 parts[2]，去掉反引号
                    raw_name = parts[2].strip().strip('`').strip()
                    features.append(raw_name)
                    in_table = True
                except (ValueError, IndexError):
                    if in_table:
                        continue
            elif in_table and len(features) >= 317:
                break  # 已收齐 317 个

    # 去重（部分特征可能重复出现）
    seen = set()
    unique = []
    for f in features:
        if f not in seen and f.strip():
            seen.add(f)
            unique.append(f.strip())

    return unique


def _extract_section(content: str, heading: str) -> Optional[str]:
    """提取 Markdown 中某个 ## 标题下的全部内容"""
    # 查找 "## {heading}"
    pattern = rf'^##\s+{re.escape(heading)}'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return None

    start = match.start()
    # 查找下一个同级 ## 标题
    next_section = re.search(r'^##\s+', content[start + 1:], re.MULTILINE)
    if next_section:
        end = start + 1 + next_section.start()
    else:
        end = len(content)

    return content[start:end]


# ---- 缓存 ----
_cache: Optional[List[str]] = None


def cached_effective_features(refresh: bool = False) -> List[str]:
    """缓存的 317 特征列表（避免重复读文件）"""
    global _cache
    if _cache is None or refresh:
        _cache = get_effective_features()
    return _cache


# ================================================================
# 自测
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    features = get_effective_features()
    print(f"全部有效特征: {len(features)} 个")
    print(f"前10: {features[:10]}")

    top50 = get_top_features(50)
    print(f"\nTop-50: {len(top50)} 个")
    print(f"前5: {top50[:5]}")
    print(f"后5: {top50[-5:]}")
