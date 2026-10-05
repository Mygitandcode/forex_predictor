"""
matplotlib 中文字体配置

在项目中任何使用 matplotlib 的地方，只需导入本模块即可自动配置中文字体：

    import config.matplotlib_fonts  # noqa: F401 — 自动注册中文字体

默认使用微软雅黑（Windows 自带），Linux/Docker 环境下会尝试备选字体。
"""

import os
import glob
import platform
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---- 字体缓存标记（避免重复注册） ----
_MATPLOTLIB_FONT_CONFIGURED = False


def configure_chinese_font():
    """查找并注册中文字体到 matplotlib，全局只执行一次。"""
    global _MATPLOTLIB_FONT_CONFIGURED
    if _MATPLOTLIB_FONT_CONFIGURED:
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        logger.debug("matplotlib 未安装，跳過字体配置")
        return

    # ---- 1. 按优先级查找字体文件 ----
    font_candidates = _find_chinese_fonts()
    registered_family = None

    for font_path in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            # 注册字体到 matplotlib（addfont 在 3.2+ 自动更新内部缓存）
            fm.fontManager.addfont(str(font_path))

            # 从字体文件中提取 family name
            props = fm.FontProperties(fname=str(font_path))
            family_name = props.get_name()
            logger.info(f"已注册字体: {family_name} ← {font_path}")

            registered_family = family_name
            break
        except Exception:
            logger.warning(f"字体注册失败: {font_path}", exc_info=True)

    # ---- 2. 配置 matplotlib 全局字体 ----
    if registered_family:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [registered_family]
        logger.info(f"matplotlib 默认字体已设为: {registered_family}")
    else:
        logger.warning("未找到任何中文字体，图表中的中文可能显示为方块")

    # ---- 3. 通用修复：解决负号显示问题 ----
    plt.rcParams["axes.unicode_minus"] = False

    _MATPLOTLIB_FONT_CONFIGURED = True


def _find_chinese_fonts():
    """按优先级返回系统中可能的中文字体路径列表。"""
    system = platform.system()

    if system == "Windows":
        return _find_windows_chinese_fonts()
    elif system == "Darwin":
        return _find_mac_chinese_fonts()
    else:
        return _find_linux_chinese_fonts()


def _find_windows_chinese_fonts():
    """Windows 中文字体（按优先级）。"""
    font_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    candidates = [
        font_dir / "msyh.ttc",       # 微软雅黑
        font_dir / "msyhbd.ttc",     # 微软雅黑粗体
        font_dir / "msyhl.ttc",      # 微软雅黑 Light
        font_dir / "simhei.ttf",     # 黑体
        font_dir / "simsun.ttc",     # 宋体
        font_dir / "simkai.ttf",     # 楷体
    ]
    # 也尝试 glob 匹配
    found = []
    for path in candidates:
        if path.exists():
            found.append(str(path))
    if not found:
        # fallback: glob 搜索任意中文字体
        found = glob.glob(str(font_dir / "msyh*")) + glob.glob(str(font_dir / "sim*"))
    return found


def _find_mac_chinese_fonts():
    """macOS 中文字体。"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    return [p for p in candidates if os.path.exists(p)]


def _find_linux_chinese_fonts():
    """Linux 中文字体（含 Docker 环境）。"""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]
    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        # glob fallback
        existing = (
            glob.glob("/usr/share/fonts/**/*wqy*", recursive=True)
            + glob.glob("/usr/share/fonts/**/*CJK*", recursive=True)
            + glob.glob("/usr/share/fonts/**/*Noto*Sans*", recursive=True)
        )
    return existing


# ---- 模块导入时自动执行（惰性触发） ----
# 为避免在非图形环境下导入 matplotlib 导致错误，
# 这里不做自动调用。使用时请显式调用 configure_chinese_font()。
__all__ = ["configure_chinese_font"]
