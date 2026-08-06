"""行情数据源抽象接口。所有方法返回 pandas.DataFrame。"""
from abc import ABC, abstractmethod

CAP_STOCK_BASIC = "stock_basic"
CAP_DAILY_PRICE = "daily_price"
CAP_REALTIME_PRICE = "realtime_price"
CAP_DAILY_BASIC = "daily_basic"
CAP_INCOME_STATEMENT = "income_statement"
CAP_INDEX_DAILY = "index_daily"
CAP_TRADE_CALENDAR = "trade_calendar"

CAPABILITY_LABELS = {
    CAP_STOCK_BASIC: "股票基础信息",
    CAP_DAILY_PRICE: "日线行情",
    CAP_REALTIME_PRICE: "实时行情",
    CAP_DAILY_BASIC: "每日指标",
    CAP_INCOME_STATEMENT: "利润表",
    CAP_INDEX_DAILY: "指数行情",
    CAP_TRADE_CALENDAR: "交易日历",
}


class CapabilityNotSupported(Exception):
    """当前数据源不支持所请求的能力。"""

    def __init__(self, provider_name, capability):
        label = CAPABILITY_LABELS.get(capability, capability)
        super().__init__(f"当前数据源（{provider_name}）不支持{label}")
        self.provider_name = provider_name
        self.capability = capability


class MarketDataProvider(ABC):
    name = "base"
    CAPABILITIES = set()

    def require(self, capability):
        """在调用具体实现前做前置能力检查。"""
        if capability not in self.CAPABILITIES:
            raise CapabilityNotSupported(self.name, capability)

    @abstractmethod
    def get_stock_basic(self, ts_code=None, name=None):
        """返回列：ts_code, symbol, name, area, industry, market, list_date"""

    @abstractmethod
    def get_daily_price(self, ts_code, start_date=None, end_date=None, adj=None):
        """返回 normalize.DAILY_COLUMNS。adj 取 None / 'qfq' / 'hfq'。"""

    @abstractmethod
    def get_realtime_price(self, ts_code):
        """ts_code 为单个代码或代码列表，返回 normalize.REALTIME_COLUMNS。"""

    @abstractmethod
    def get_daily_basic(self, ts_code, start_date=None, end_date=None):
        """返回列：ts_code, trade_date, pe, pe_ttm, pb, ps_ttm, dv_ratio, total_mv, circ_mv"""

    @abstractmethod
    def get_income_statement(self, ts_code, start_date=None, end_date=None):
        """返回列：ts_code, end_date, total_revenue, revenue, operate_profit, total_profit, n_income"""

    @abstractmethod
    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        """返回 normalize.DAILY_COLUMNS。"""

    @abstractmethod
    def get_trade_calendar(self, start_date=None, end_date=None):
        """返回列：cal_date（YYYYMMDD 字符串），仅含交易日。"""
