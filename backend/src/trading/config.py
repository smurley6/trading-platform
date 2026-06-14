"""Environment-based configuration, validated at startup (spec sections 7.3, 16).

Risk caps have no defaults on purpose: the process refuses to start without them.
"""

from decimal import Decimal

from pydantic_settings import BaseSettings, SettingsConfigDict

from trading.domain.sleeves import TradingMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mode_default: TradingMode = TradingMode.PAPER

    # Event store: in-memory when unset (dev), Postgres when set.
    database_url: str | None = None

    # Sleeve / paper account
    starting_cash: Decimal = Decimal("100000")

    # Conviction (spec section 16)
    trade_threshold: float = 70.0
    watch_threshold: float = 40.0
    corroboration_window_days: int = 21

    # Risk caps — required at startup, no defaults (FR-RISK-2)
    max_trade_pct_of_sleeve: Decimal
    max_position_concentration_pct: Decimal
    max_drawdown_pct: Decimal

    # Live trading (M5, spec sections 11 & 13). Live is OFF by construction: the master
    # switch defaults false, live credentials are unset, and a sleeve can only go live by
    # passing the readiness preflight. The configured risk caps above MUST be at or under
    # these ceilings before any sleeve may opt in (risk hardening).
    allow_live: bool = False
    alpaca_live_key_id: str | None = None
    alpaca_live_secret_key: str | None = None
    alpaca_live_base_url: str = "https://api.alpaca.markets"
    live_max_trade_pct_of_sleeve: Decimal = Decimal("5")
    live_max_position_concentration_pct: Decimal = Decimal("15")
    live_max_drawdown_pct: Decimal = Decimal("10")

    # Live market data feed (live stock tracking + live P&L marks). OFF by construction: a
    # bare `python -m trading.main` injects the disabled feed and makes no network calls.
    # The feed turns on only when `enable_live_quotes` is true AND data credentials resolve
    # (see `alpaca_data_key_id`/`alpaca_data_secret_key` below). Alpaca's data API accepts
    # either paper or live keys, so by default the feed reuses the live broker credentials if
    # present; the dedicated `alpaca_data_*` vars override that (env-only, never committed).
    enable_live_quotes: bool = False
    alpaca_data_base_url: str = "https://data.alpaca.markets"
    # Real-time quote stream behind /ws/quotes. Same ENABLE_LIVE_QUOTES + data-cred switch as
    # the REST feed gates it — no separate on/off flag. Dormant by default: the disabled stream
    # opens no socket; the live WebSocket is dialed only when the feed is enabled and creds
    # resolve (see main.py). Override only to point at a different Alpaca data feed (e.g. SIP).
    alpaca_data_stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    alpaca_data_key_id: str | None = None
    alpaca_data_secret_key: str | None = None
    # Watchlist used by /quotes/tracked when the platform holds no positions yet.
    default_watchlist: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL"]

    # Real historical bars for backtests (FR-BT). OFF by construction: a bare
    # `python -m trading.main` and the test suite price backtests off the deterministic
    # synthetic path and make ZERO network calls. The backtest fetches REAL daily OHLC bars
    # (bar close per day) from the Alpaca data API only when `enable_historical_data` is true
    # AND data credentials resolve (the same `data_key_id()`/`data_secret_key()` + base URL as
    # the live quote feed). `backtest_lookback_days` bounds the bar date range when the recorded
    # signals do not already span one.
    enable_historical_data: bool = False
    backtest_lookback_days: int = 180

    # Alpaca PAPER trading venue (FR-EXE-1, FR-REC-1). A REAL paper account — no real money,
    # distinct from the live venue. When enabled (and paper or data credentials resolve), the
    # `alpaca_paper_sleeve_id` sleeve executes against your Alpaca paper account instead of the
    # in-process simulator, so reconciliation diffs local state against an authoritative
    # external broker. OFF by construction: disabled -> the in-memory PaperBroker, no network.
    # One Alpaca account is one book, so exactly ONE sleeve routes here (others stay simulated)
    # to keep per-sleeve reconciliation meaningful.
    enable_alpaca_paper: bool = False
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_paper_key_id: str | None = None
    alpaca_paper_secret_key: str | None = None
    alpaca_paper_sleeve_id: str = "sleeve-political"

    def data_key_id(self) -> str | None:
        """Resolved market-data key: the dedicated data key, else the live broker key."""
        return self.alpaca_data_key_id or self.alpaca_live_key_id

    def data_secret_key(self) -> str | None:
        return self.alpaca_data_secret_key or self.alpaca_live_secret_key

    def paper_key_id(self) -> str | None:
        """Paper trading key: the dedicated paper key, else the data key — an Alpaca paper key
        pair serves both trading and data, so reusing the data key 'just works'."""
        return self.alpaca_paper_key_id or self.alpaca_data_key_id

    def paper_secret_key(self) -> str | None:
        return self.alpaca_paper_secret_key or self.alpaca_data_secret_key

    # Data sources (FR-ING). External sources default OFF so a bare `python -m trading.main`
    # makes no network calls; configure these to enable the political pipeline.
    news_feed_urls: list[str] = []
    usaspending_enabled: bool = False
    usaspending_lookback_days: int = 30
    disclosed_trade_leg_enabled: bool = False
    congress_trades_endpoint: str | None = None
    fmp_api_key: str | None = None
    http_timeout_seconds: float = 10.0

    # Jobs
    reconcile_interval_seconds: int = 300
    # 0 disables the scheduled political cycle; set > 0 to poll sources on an interval.
    ingest_interval_seconds: int = 0

    log_level: str = "INFO"
