"""events table: append-only system of record, immutability enforced by trigger

Revision ID: 0001
Revises:
Create Date: 2026-06-12

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=True), nullable=False, unique=True),
        sa.Column("stream_id", sa.Text(), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=False),
        sa.UniqueConstraint("stream_id", "seq", name="uq_events_stream_seq"),
    )
    op.create_index("ix_events_stream_id", "events", ["stream_id"])
    op.create_index("ix_events_correlation_id", "events", ["correlation_id"])
    op.create_index("ix_events_event_type", "events", ["event_type"])

    # Spec section 9: append-only by construction. UPDATE/DELETE fail at the database
    # level regardless of application code or ad-hoc SQL sessions.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION prevent_event_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'events table is append-only: % is forbidden', TG_OP;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER events_append_only
        BEFORE UPDATE OR DELETE ON events
        FOR EACH ROW EXECUTE FUNCTION prevent_event_mutation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER events_no_truncate
        BEFORE TRUNCATE ON events
        FOR EACH STATEMENT EXECUTE FUNCTION prevent_event_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS events_no_truncate ON events")
    op.execute("DROP TRIGGER IF EXISTS events_append_only ON events")
    op.execute("DROP FUNCTION IF EXISTS prevent_event_mutation()")
    op.drop_table("events")
