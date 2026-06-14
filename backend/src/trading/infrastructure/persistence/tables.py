"""SQLAlchemy table definitions.

The `events` table is the append-only system of record. The `rm_*` tables and the
`portfolio_snapshots` hypertable are CQRS read-side projections — current-state views
rebuildable from the log (spec section 9), kept consistent with each append.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

events_table = Table(
    "events",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),  # global recording order
    Column("event_id", Uuid(as_uuid=True), nullable=False, unique=True),
    Column("stream_id", Text, nullable=False),
    Column("seq", BigInteger, nullable=False),  # gap-free per stream
    Column("event_type", Text, nullable=False),
    Column("schema_version", Integer, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("payload", JSONB, nullable=False),
    Column("correlation_id", Text, nullable=False),
    UniqueConstraint("stream_id", "seq", name="uq_events_stream_seq"),
    Index("ix_events_stream_id", "stream_id"),
    Index("ix_events_correlation_id", "correlation_id"),
    Index("ix_events_event_type", "event_type"),
)

# --- Read-model (CQRS read side) tables. Rebuildable from `events`. ---

# Reconstructed signals, one row per recorded `signal.received`. Indexed on created_at so the
# rolling corroboration window is an indexed range scan, not a full-log replay.
rm_signals_table = Table(
    "rm_signals",
    metadata,
    Column("signal_id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("recorded_id", BigInteger, nullable=False),  # events.id, for stable feed ordering
    Column("source_type", Text, nullable=False),
    Column("ticker", Text, nullable=False),
    Column("politician_id", Text, nullable=True),
    Column("sentiment", Numeric, nullable=False),
    Column("politician_weight", Numeric, nullable=False),
    Column("source_reliability", Numeric, nullable=False),
    Column("raw_payload", JSONB, nullable=False),
    Index("ix_rm_signals_created_at", "created_at"),
    Index("ix_rm_signals_recorded_id", "recorded_id"),
)

# Current open positions with cost basis, one row per (sleeve, symbol). Closed lots are
# deleted (zero quantity), mirroring the in-memory projection.
rm_positions_table = Table(
    "rm_positions",
    metadata,
    Column("sleeve_id", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("quantity", Numeric, nullable=False),
    Column("avg_price", Numeric, nullable=False),
    PrimaryKeyConstraint("sleeve_id", "symbol", name="pk_rm_positions"),
)

# Latest known local status per client order.
rm_order_statuses_table = Table(
    "rm_order_statuses",
    metadata,
    Column("client_order_id", Text, primary_key=True),
    Column("sleeve_id", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False),
)

# Equity-curve time-series (TimescaleDB hypertable; see migration 0002). Append-only points;
# "latest per sleeve" is a DISTINCT ON query.
portfolio_snapshots_table = Table(
    "portfolio_snapshots",
    metadata,
    Column("sleeve_id", Text, nullable=False),
    Column("taken_at", DateTime(timezone=True), nullable=False),
    Column("cash", Numeric, nullable=False),
    Column("equity", Numeric, nullable=False),
    Index("ix_portfolio_snapshots_sleeve_taken", "sleeve_id", "taken_at"),
)
