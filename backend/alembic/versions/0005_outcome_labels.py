"""add outcome_labels research projection table

Task 03 (docs/RESEARCH-PLAN.md §2-§3): the outcome labeler computes forward event-study abnormal
returns for ticker-resolved signals and stores them here. These labels are DERIVED, not truth —
they live in a REBUILDABLE PROJECTION, never the immutable event log. Rebuild semantics are
delete-then-write by label_run_id (see application/outcome_labels.py::OutcomeLabelsPort), so
label_run_id is indexed; signal_id is indexed for the Task-05 diagnostic's per-signal queries.

One row per (signal, horizon). abnormal_return is a double precision float, nullable (null for
"unresolvable" rows). A nullable `reason` records WHY an unresolvable label could not be computed
(insufficient_forward_bars / no_bars / benchmark_bars_missing, …) — the spec requires
unresolvable labels carry a reason, and it must be queryable, not just logged. `benchmark`
records the proxy used ("sector:<ETF>" for a sector proxy, "market" for the SPY fallback) so the
diagnostic can split results on the prefix.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-02

"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "outcome_labels",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("signal_id", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("abnormal_return", sa.Double(), nullable=True),
        sa.Column("benchmark", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("label_run_id", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_outcome_labels_label_run_id", "outcome_labels", ["label_run_id"]
    )
    op.create_index("ix_outcome_labels_signal_id", "outcome_labels", ["signal_id"])


def downgrade() -> None:
    op.drop_index("ix_outcome_labels_signal_id", table_name="outcome_labels")
    op.drop_index("ix_outcome_labels_label_run_id", table_name="outcome_labels")
    op.drop_table("outcome_labels")
