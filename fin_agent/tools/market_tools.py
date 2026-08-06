"""统一的行情工具层：调用 provider、序列化结果、把不支持的能力转为中文提示。"""
import functools

from fin_agent.config import Config
from fin_agent.datasources import CapabilityNotSupported, get_provider

TUSHARE_REQUIRED_HINT = (
    "该功能需要 Tushare Token 才能使用。请打开设置页填写 Tushare Token 后重试，"
    "或改用当前数据源已支持的功能。"
)


def df_to_json(df):
    if df is None or len(df) == 0:
        return "[]"
    return df.to_json(orient="records", force_ascii=False, date_format="iso")


def _guard(func):
    """把 provider 抛出的异常转为可直接回给模型的中文文本。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CapabilityNotSupported as e:
            return f"{e}。可在设置页切换数据源后重试。"
        except Exception as e:
            return f"Error: {e}"
    return wrapper


def requires_tushare(func):
    """标记仅 Tushare 可用的工具：无 Token 时直接返回引导语，不调用底层接口。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not Config.TUSHARE_TOKEN:
            return TUSHARE_REQUIRED_HINT
        return func(*args, **kwargs)
    return wrapper


@_guard
def get_stock_basic(ts_code=None, name=None):
    return df_to_json(get_provider().get_stock_basic(ts_code=ts_code, name=name))


@_guard
def get_daily_price(ts_code, start_date=None, end_date=None, adj=None):
    return df_to_json(get_provider().get_daily_price(ts_code, start_date, end_date, adj))


@_guard
def get_realtime_price(ts_code):
    return df_to_json(get_provider().get_realtime_price(ts_code))


@_guard
def get_daily_basic(ts_code, start_date=None, end_date=None):
    return df_to_json(get_provider().get_daily_basic(ts_code, start_date, end_date))


@_guard
def get_income_statement(ts_code, start_date=None, end_date=None):
    return df_to_json(get_provider().get_income_statement(ts_code, start_date, end_date))


@_guard
def get_index_daily(ts_code, start_date=None, end_date=None):
    return df_to_json(get_provider().get_index_daily(ts_code, start_date, end_date))
