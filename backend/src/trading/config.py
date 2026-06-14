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
