"""make rm_signals.ticker nullable (unresolved-entity backfilled signals)

Task 04b (docs/RESEARCH-PLAN.md): the House disclosures historical fetcher runs the entity
resolver on a PTR's asset name when no ticker regex match is found, and — per the research
plan's survivorship-awareness constraint — emits the signal anyway with ticker=None rather than
dropping it (the unresolved set is itself research data). `domain/signals.py::Signal.ticker`
is now `str | None`; this migration relaxes the previously-NOT-NULL `rm_signals.ticker` column
to match, so the read model can round-trip an unresolved signal instead of failing the INSERT.

Unresolved (None-ticker) signals stay visible in feeds/projections but are excluded from
conviction scoring, strategy evaluation, proposals, backtest symbol collection, and outcome
labeling — every ticker-grouping site filters `is not None` first (see MEMORY.md 2026-07-02).

`outcome_labels.ticker` is UNCHANGED (stays NOT NULL): the labeler only ever labels
ticker-resolved signals, so a label row can never carry a null ticker.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-02

"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("rm_signals", "ticker", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    # Any existing NULL tickers must be backfilled/removed before a real rollback could succeed;
    # this mirrors the other migrations' unconditional downgrade (no data massaging attempted).
    op.alter_column("rm_signals", "ticker", existing_type=sa.Text(), nullable=False)
