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

    # Conviction scoring tunables (FR-POL-4/5, spec section 12). Defaults reproduce the M1
    # formula EXACTLY (decay off => equal weights, min-sources=1, bonus 0.25, scale 25), so
    # platform behavior is unchanged until you deliberately tune them. Time decay weights more
    # recent signals more heavily within the corroboration window (half-life in days; <= 0
    # disables decay). `min_sources_for_full_corroboration` raises the bar for high conviction;
    # `max_corroboration_factor` caps the multi-source bonus. This is the core algorithm surface.
    conviction_decay_half_life_days: float = 0.0
    conviction_corroboration_bonus_per_source: float = 0.25
    conviction_min_sources_for_full_corroboration: int = 1
    conviction_max_corroboration_factor: float | None = None
    conviction_scale: float = 25.0

    # Risk caps — required at startup, no defaults (FR-RISK-2)
    max_trade_pct_of_sleeve: Decimal
    max_position_concentration_pct: Decimal
    max_drawdown_pct: Decimal

    # Data integrity (anti-placeholder). When True the platform refuses to let SYNTHETIC / fake /
    # placeholder / seeded data reach a real decision: synthetic signals are dropped at ingestion,
    # a real-broker evaluation aborts on any non-REAL signal, and a real sleeve will not boot
    # without real market data. The composition root turns this ON AUTOMATICALLY whenever a real
    # Alpaca account is wired (ENABLE_ALPACA_PAPER or ALLOW_LIVE) — you cannot trade a real account
    # on fakes — and you may set it True explicitly to enforce the same on the in-process
    # simulator. Tests and backtests run with it OFF (synthetic by design, never a real broker).
    require_real_data: bool = False

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
    # Real congressional roster (FR-POL-3). When true, the unitedstates/congress-legislators
    # dataset is fetched at startup so news / disclosed-trade signals attach to REAL politicians
    # (Provenance.REAL) and can pass the anti-placeholder guard. Off => the synthetic seed roster,
    # whose signals are SYNTHETIC and blocked from real decisions. Only fetched when a
    # politician-using source (news or disclosed trades) is enabled.
    enable_congress_roster: bool = False
    usaspending_enabled: bool = False
    usaspending_lookback_days: int = 30
    disclosed_trade_leg_enabled: bool = False
    congress_trades_endpoint: str | None = None
    fmp_api_key: str | None = None

    # Official free government-disclosure adapters (FR-ING-2). All OFF by default. House/Senate
    # PTRs attach to the REAL congress roster (need ENABLE_CONGRESS_ROSTER) and are lagged
    # corroboration. SEC Form 4 (corporate insider), LDA lobbying, and openFDA approvals stand
    # alone (no politician). Every emitted Signal is stamped Provenance.REAL.
    house_disclosures_enabled: bool = False
    house_disclosures_lookback_days: int = 30
    house_disclosures_max_filings: int = 50
    senate_efd_enabled: bool = False
    senate_efd_lookback_days: int = 30
    sec_form4_enabled: bool = False
    sec_form4_user_agent: str | None = None  # REQUIRED when enabled (SEC 403s without a UA)
    sec_form4_lookback_days: int = 3
    lobbying_enabled: bool = False
    lobbying_lookback_days: int = 30
    lda_api_key: str | None = None
    fda_approvals_enabled: bool = False
    fda_lookback_days: int = 30
    openfda_api_key: str | None = None

    http_timeout_seconds: float = 10.0

    # Jobs
    reconcile_interval_seconds: int = 300
    # 0 disables the scheduled political cycle; set > 0 to poll sources on an interval.
    ingest_interval_seconds: int = 0
    # Order sync (FR-REC, FR-EXE): pull the broker's order state and emit the lifecycle events
    # the local log is missing — closes the loop for the async-filling Alpaca venue. 0 disables
    # the scheduled job (the default); set > 0 (seconds) to run it on an interval. Harmless with
    # the in-process PaperBroker (fills are already recorded synchronously, so it emits nothing).
    order_sync_interval_seconds: int = 0

    log_level: str = "INFO"

    # --- Hardening: auth (interface/composition agent) --------------------------------------
    # API key guarding the MUTATING/sensitive endpoints (POST /signals, /ingest, /backtest,
    # /orders/sync, /orders/{id}/cancel, /reconcile, /kill-switch, /sleeves/{id}/mode and
    # /sleeves/{id}/readiness). UNSET (the default) => those endpoints stay OPEN, and the app
    # logs a single startup warning so an unprotected deployment is never silent. SET => callers
    # must present it via `Authorization: Bearer <key>` or the `X-API-Key` header; a mismatch is
    # 401. GET read endpoints stay open regardless. Env-only, NEVER commit a real key.
    api_key: str | None = None

    # --- Hardening: observability (interface/composition agent) -----------------------------
    # Structured JSON request logging + Prometheus-text /metrics are always on (dependency-light,
    # hand-rolled). This flag only toggles the /metrics endpoint's exposure should an operator
    # want to hide it (e.g. behind a separate scrape network). Liveness (/healthz) and readiness
    # (/readyz) are always present.
    enable_metrics: bool = True

    # --- Hardening: real-time trade-updates stream (interface/composition agent) -------------
    # Alpaca paper trade-updates WebSocket (account stream): records fills/status changes as
    # events in REAL TIME, complementing the 30s polling order-sync (which stays as the
    # reconciliation fallback). OFF by construction: even with ENABLE_ALPACA_PAPER on, this is
    # only dialed when ENABLE_TRADE_UPDATES_STREAM=true AND paper credentials resolve. A bare run
    # opens no socket and makes no network call. The stream URL is derived from
    # `alpaca_paper_base_url` (https -> wss, + /stream) unless overridden here.
    enable_trade_updates_stream: bool = False
    alpaca_trade_updates_stream_url: str | None = None

    def trade_updates_stream_url(self) -> str:
        """Resolved trade-updates WebSocket URL: explicit override, else derived from the paper
        base URL (the paper `https://...` host becomes `wss://<host>/stream`)."""
        if self.alpaca_trade_updates_stream_url:
            return self.alpaca_trade_updates_stream_url
        base = self.alpaca_paper_base_url.rstrip("/")
        scheme = "wss" if base.startswith("https") else "ws"
        host = base.split("://", 1)[-1]
        return f"{scheme}://{host}/stream"
