"""SQLAlchemy table definitions.

The `events` table is the append-only system of record. The `rm_*` tables and the
`portfolio_snapshots` hypertable are CQRS read-side projections — current-state views
rebuildable from the log (spec section 9), kept consistent with each append.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Double,
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
    # Nullable since migration 0006: an unresolved-entity backfilled signal (Task 04b) is
    # recorded with ticker=None rather than dropped. See domain/signals.py::Signal.ticker.
    Column("ticker", Text, nullable=True),
    Column("politician_id", Text, nullable=True),
    Column("sentiment", Numeric, nullable=False),
    Column("politician_weight", Numeric, nullable=False),
    Column("source_reliability", Numeric, nullable=False),
    Column("raw_payload", JSONB, nullable=False),
    # Added in migration 0003. Server default is fail-closed ('synthetic'/'exact') so
    # pre-existing rows (recorded before these columns existed) never silently upgrade to a
    # trusted tier; a read-model rebuild from the event log restores their true values.
    Column("provenance", Text, nullable=False, server_default="synthetic"),
    Column("pit_confidence", Text, nullable=False, server_default="exact"),
    Index("ix_rm_signals_created_at", "created_at"),
    Index("ix_rm_signals_recorded_id", "recorded_id"),
)

# Backfill dedupe (Task 02): the source-native ids already ingested by a backfill run, keyed on
# (source_id, source_uid). Consulted BEFORE appending signal.received events so re-running a
# window double-appends nothing (idempotency at the event-log layer, not just the read model).
# Rebuildable from the log: source_uid is also carried in each signal's raw_payload.
backfill_seen_table = Table(
    "backfill_seen",
    metadata,
    Column("source_id", Text, nullable=False),
    Column("source_uid", Text, nullable=False),
    PrimaryKeyConstraint("source_id", "source_uid", name="pk_backfill_seen"),
)

# Outcome labels (Task 03): a rebuildable RESEARCH projection of event-study outcome labels, one
# row per (signal, horizon). NOT part of the event-fold read model — it is written by an explicit
# labeler run (see application/outcome_labels.py) and rebuilt by re-running the labeler, keyed and
# indexed on label_run_id so a rerun can delete-then-write its own run. abnormal_return is a
# double (nullable — null when unresolvable); reason is nullable (set only for unresolvable rows).
outcome_labels_table = Table(
    "outcome_labels",
    metadata,
    Column(
        "id", BigInteger, Identity(), primary_key=True
    ),  # surrogate key; (signal,horizon) repeats across runs
    Column("signal_id", Text, nullable=False),
    Column("ticker", Text, nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("abnormal_return", Double, nullable=True),
    Column("benchmark", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("reason", Text, nullable=True),
    Column("label_run_id", Text, nullable=False),
    Index("ix_outcome_labels_label_run_id", "label_run_id"),
    Index("ix_outcome_labels_signal_id", "signal_id"),
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
