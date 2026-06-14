"""SQLAlchemy table definitions. The `events` table is the append-only system of record."""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
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
