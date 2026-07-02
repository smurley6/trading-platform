"""add backfill_seen dedupe table

Task 02 (docs/RESEARCH-PLAN.md): the backfill job records historical signals idempotently. Its
dedupe key is the source-native id ((source_id, source_uid) — EDGAR accession number, House
DocID + row, …), checked BEFORE any signal.received event is appended so re-running a window
double-appends nothing (idempotency at the EVENT-LOG layer, not just the read model).

This is a small DEDICATED table rather than a widening of rm_signals: it keeps the read model's
shape untouched and makes the seen-check an O(1) primary-key lookup on the source-native id.
Because each backfilled signal also carries its source_uid inside raw_payload, the table is
fully rebuildable from the append-only event log.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-01

"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "backfill_seen",
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_uid", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("source_id", "source_uid", name="pk_backfill_seen"),
    )


def downgrade() -> None:
    op.drop_table("backfill_seen")
