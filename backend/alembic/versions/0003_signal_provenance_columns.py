"""add provenance + pit_confidence columns to rm_signals

Task 01 (docs/RESEARCH-PLAN.md): the Postgres read model reconstructed Signal objects without
provenance/pit_confidence, so everything read back from `rm_signals` silently degraded to
SYNTHETIC/"exact" regardless of what was recorded in the event log. This adds the two columns
so the read side round-trips what `signal.received` events actually carry.

Server defaults are fail-closed for pre-existing rows: 'synthetic' (never mistaken for real)
and 'exact' (matches the pre-backfill assumption that all recorded signals were live-timestamped).
Pre-existing prod rows will show 'synthetic' until a full read-model rebuild from the event log
restores their true provenance (most were already REAL going in, since the anti-placeholder
guard keeps SYNTHETIC out of real accounts — the column was simply never populated before now).

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-01

"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rm_signals",
        sa.Column("provenance", sa.Text(), nullable=False, server_default="synthetic"),
    )
    op.add_column(
        "rm_signals",
        sa.Column("pit_confidence", sa.Text(), nullable=False, server_default="exact"),
    )


def downgrade() -> None:
    op.drop_column("rm_signals", "pit_confidence")
    op.drop_column("rm_signals", "provenance")
