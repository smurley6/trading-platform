"""Task 12: Form 4 ownership-XML enrichment — reportingOwnerRelationship role flags + aff10b5One.

Data-plumbing ONLY: the parser now threads ``is_officer`` / ``is_director`` /
``is_ten_percent_owner`` / ``officer_title`` / ``is_10b5_1`` from the ownership document into every
emitted signal's ``raw_payload`` (both the live and historical builders), so role and 10b5-1 plan
splits become computable after a future re-backfill. NO conviction/weight/filter depends on these
flags — these tests assert the plumbing and its null semantics, nothing about scoring.

Every flag is nullable so "absent in the source XML" stays distinguishable from a genuine False;
the 10b5-1 flag in particular is never False-by-default.
"""

from datetime import UTC, datetime
from pathlib import Path

from trading.infrastructure.data_sources.sec_form4 import (
    _classify_transactions,
    _historical_signal,
    _live_signal,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "sec"

_RELATIONSHIP_KEYS = (
    "is_officer",
    "is_director",
    "is_ten_percent_owner",
    "officer_title",
    "is_10b5_1",
)


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _only(document: str):  # type: ignore[no-untyped-def]
    parsed = _classify_transactions(document)
    assert len(parsed) == 1, "fixtures carry exactly one open-market (P) row"
    return parsed[0]


# --- Acceptance test 1: director-only ----------------------------------------------------------


def test_director_only_sets_is_director_true_others_false_none() -> None:
    txn = _only(_read("form4_rel_director_only.xml"))
    assert txn.is_director is True
    assert txn.is_officer is False
    assert txn.is_ten_percent_owner is False
    # officerTitle absent in a director-only filing -> None, never "".
    assert txn.officer_title is None
    # No aff10b5One element in this fixture -> None (never False-by-default).
    assert txn.is_10b5_1 is None


def test_officer_title_captured_verbatim() -> None:
    txn = _only(_read("form4_rel_officer_title.xml"))
    assert txn.is_officer is True
    assert txn.is_director is False
    assert txn.is_ten_percent_owner is False
    assert txn.officer_title == "Chief Financial Officer"


def test_ten_percent_owner_flag() -> None:
    txn = _only(_read("form4_rel_ten_percent.xml"))
    assert txn.is_ten_percent_owner is True
    assert txn.is_officer is False
    assert txn.is_director is False
    assert txn.officer_title is None


def test_combined_flags_all_true_with_title() -> None:
    txn = _only(_read("form4_rel_combined.xml"))
    assert txn.is_officer is True
    assert txn.is_director is True
    assert txn.is_ten_percent_owner is True
    assert txn.officer_title == "President and COO"
    assert txn.is_10b5_1 is True


# --- Acceptance test 2: aff10b5One true / false / absent ----------------------------------------


def test_aff10b5one_word_true_yields_true() -> None:
    assert _only(_read("form4_10b5_true_word.xml")).is_10b5_1 is True


def test_aff10b5one_numeric_one_yields_true() -> None:
    # Combined fixture uses the numeric "1" form.
    assert _only(_read("form4_rel_combined.xml")).is_10b5_1 is True


def test_aff10b5one_zero_yields_false() -> None:
    assert _only(_read("form4_10b5_false.xml")).is_10b5_1 is False


def test_aff10b5one_absent_yields_none_never_false() -> None:
    txn = _only(_read("form4_10b5_absent.xml"))
    assert txn.is_10b5_1 is None
    assert txn.is_10b5_1 is not False


# --- Acceptance test 3: missing/malformed relationship block is backward-compatible -------------


def test_missing_relationship_block_flags_all_none_but_signal_still_emitted() -> None:
    parsed = _classify_transactions(_read("form4_rel_missing.xml"))
    # The signal is still emitted (backward compatible) even with no relationship block.
    assert len(parsed) == 1
    txn = parsed[0]
    assert txn.is_officer is None
    assert txn.is_director is None
    assert txn.is_ten_percent_owner is None
    assert txn.officer_title is None
    assert txn.is_10b5_1 is None


def test_malformed_relationship_block_does_not_crash() -> None:
    # A relationship block present but with a broken/unexpected boolean value must not raise and
    # must degrade to None rather than a coerced False.
    document = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>maybe</isOfficer>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-06-11</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10</value></transactionShares>
                <transactionPricePerShare><value>1.0</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""
    txn = _only(document)
    # Unrecognized boolean text -> None (tri-state), not False.
    assert txn.is_officer is None
    assert txn.is_director is None
    assert txn.is_10b5_1 is None


# --- Payload emission: BOTH builders carry the new keys, with matching null semantics -----------


def test_live_payload_carries_relationship_keys() -> None:
    txn = _only(_read("form4_rel_combined.xml"))
    payload = _live_signal(txn, source_url="http://example/x.xml").raw_payload
    for key in _RELATIONSHIP_KEYS:
        assert key in payload
    assert payload["is_officer"] is True
    assert payload["is_director"] is True
    assert payload["is_ten_percent_owner"] is True
    assert payload["officer_title"] == "President and COO"
    assert payload["is_10b5_1"] is True


def test_historical_payload_carries_relationship_keys_and_pit_created_at() -> None:
    txn = _only(_read("form4_rel_director_only.xml"))
    accepted_at = datetime(2026, 6, 12, 20, 30, tzinfo=UTC)
    signal = _historical_signal(
        txn,
        accession="0000320193-26-000001",
        accepted_at=accepted_at,
        source_url="http://example/x.txt",
    )
    payload = signal.raw_payload
    for key in _RELATIONSHIP_KEYS:
        assert key in payload
    assert payload["is_director"] is True
    assert payload["is_officer"] is False
    assert payload["officer_title"] is None
    assert payload["is_10b5_1"] is None
    # created_at semantics unchanged: the PIT-correct acceptance datetime, NOT the transaction date.
    assert signal.created_at == accepted_at
    assert signal.created_at.date().isoformat() != payload["transaction_date"]


def test_none_flags_survive_into_payload_as_none() -> None:
    txn = _only(_read("form4_rel_missing.xml"))
    live = _live_signal(txn, source_url="http://example/x.xml").raw_payload
    hist = _historical_signal(
        txn,
        accession="0000320193-26-000002",
        accepted_at=datetime(2026, 6, 12, tzinfo=UTC),
        source_url="http://example/x.txt",
    ).raw_payload
    for payload in (live, hist):
        for key in _RELATIONSHIP_KEYS:
            assert key in payload
            assert payload[key] is None


# --- Acceptance test 4: existing payload keys byte-identical for old fixtures -------------------


def test_existing_payload_keys_unchanged_for_apple_fixture() -> None:
    """The golden apple fixture already carries a relationship block; the existing payload keys
    (owner, transaction_code/date/value) keep their exact pre-Task-12 values — the enrichment is
    purely additive."""
    txn = _only(_read("form4_apple_purchase.xml"))
    live = _live_signal(txn, source_url="http://example/a.xml").raw_payload
    assert live["owner"] == "COOK TIMOTHY D"
    assert live["ticker"] == "AAPL"
    assert live["transaction_code"] == "P"
    assert live["transaction_date"] == "2026-06-11"
    assert live["transaction_value"] == 10000 * 195.50
    # Additive enrichment: Apple's Cook is an officer (CEO), not a director.
    assert live["is_officer"] is True
    assert live["is_director"] is False
    assert live["officer_title"] == "Chief Executive Officer"
    assert live["is_10b5_1"] is None
