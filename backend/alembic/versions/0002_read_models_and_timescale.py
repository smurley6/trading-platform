"""read-model tables + TimescaleDB hypertable for portfolio snapshots (equity curve)

CQRS read side: current-state projections derived from the append-only `events` log
(spec section 9). `portfolio_snapshots` is time-series data, stored in a TimescaleDB
hypertable per the spec's time-series-storage mandate.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-13

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    # TimescaleDB powers the equity-curve hypertable WHEN available (the CI image
    # timescale/timescaledb:latest-pg16 ships it). On a stock Postgres (e.g. the Neon / Supabase
    # free tier) the extension is absent, so we skip it and leave portfolio_snapshots a plain
    # table — fully functional, just not time-partitioned. This keeps the migration portable so
    # the scheduled deployment can run on any free Postgres.
    has_timescale = (
        bind.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).scalar()
        is not None
    )
    if has_timescale:
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    op.create_table(
        "rm_signals",
        sa.Column("signal_id", sa.Text(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_id", sa.BigInteger(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("politician_id", sa.Text(), nullable=True),
        sa.Column("sentiment", sa.Numeric(), nullable=False),
        sa.Column("politician_weight", sa.Numeric(), nullable=False),
        sa.Column("source_reliability", sa.Numeric(), nullable=False),
        sa.Column("raw_payload", JSONB(), nullable=False),
    )
    op.create_index("ix_rm_signals_created_at", "rm_signals", ["created_at"])
    op.create_index("ix_rm_signals_recorded_id", "rm_signals", ["recorded_id"])

    op.create_table(
        "rm_positions",
        sa.Column("sleeve_id", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(), nullable=False),
        sa.Column("avg_price", sa.Numeric(), nullable=False),
        sa.PrimaryKeyConstraint("sleeve_id", "symbol", name="pk_rm_positions"),
    )

    op.create_table(
        "rm_order_statuses",
        sa.Column("client_order_id", sa.Text(), primary_key=True),
        sa.Column("sleeve_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.Text(), nullable=False),
    )

    op.create_table(
        "portfolio_snapshots",
        sa.Column("sleeve_id", sa.Text(), nullable=False),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cash", sa.Numeric(), nullable=False),
        sa.Column("equity", sa.Numeric(), nullable=False),
    )
    op.create_index(
        "ix_portfolio_snapshots_sleeve_taken",
        "portfolio_snapshots",
        ["sleeve_id", "taken_at"],
    )
    # Time-series storage: convert to a hypertable when TimescaleDB is present; on stock Postgres
    # the plain table above is used as-is (the read model queries it identically).
    if has_timescale:
        op.execute(
            "SELECT create_hypertable('portfolio_snapshots', 'taken_at', if_not_exists => TRUE);"
        )


def downgrade() -> None:
    op.drop_table("portfolio_snapshots")
    op.drop_table("rm_order_statuses")
    op.drop_table("rm_positions")
    op.drop_index("ix_rm_signals_recorded_id", table_name="rm_signals")
    op.drop_index("ix_rm_signals_created_at", table_name="rm_signals")
    op.drop_table("rm_signals")
