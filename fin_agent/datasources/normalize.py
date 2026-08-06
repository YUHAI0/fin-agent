"""代码、日期与列名的归一化工具。基准格式为 Tushare 风格。"""
import re

import pandas as pd

DAILY_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "pre_close", "change", "pct_chg", "vol", "amount"
]

REALTIME_COLUMNS = ["ts_code", "name", "price", "pre_close", "change", "pct_chg", "vol", "amount"]


def to_ts_code(code):
    """把任意形式的股票代码转为 Tushare 格式，例如 600519 -> 600519.SH。

    已带后缀的原样返回（统一大写）。无法判断交易所时抛 ValueError。
    """
    if code is None:
        raise ValueError("股票代码不能为空")
    code = str(code).strip().upper()
    if "." in code:
        return code

    digits = re.sub(r"\D", "", code)
    if len(digits) != 6:
        raise ValueError(f"无法识别的股票代码：{code}")

    head = digits[0]
    if head in ("6", "5", "9"):
        suffix = "SH"
    elif head in ("0", "2", "3"):
        suffix = "SZ"
    elif head in ("4", "8"):
        suffix = "BJ"
    else:
        raise ValueError(f"无法判断交易所：{code}")
    return f"{digits}.{suffix}"


def to_plain_code(ts_code):
    """把 600519.SH 转为 600519。"""
    return str(ts_code).strip().upper().split(".")[0]


def normalize_date(value):
    """把日期归一为 YYYYMMDD 字符串。接受 date/datetime/2024-01-05/20240105。"""
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    text = re.sub(r"\D", "", str(value))
    return text[:8] if len(text) >= 8 else text


def rename_and_select(df, mapping, columns):
    """按 mapping 重命名列，补齐 columns 中缺失的列为 None，再按 columns 顺序返回。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=columns)
    out = df.rename(columns=mapping)
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns]
