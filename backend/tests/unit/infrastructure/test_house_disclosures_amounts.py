"""Amount-bracket parsing tests for the House Clerk PTR adapter (Task 07, Scope B).

``_parse_amount_range``/``_parse_transactions`` are pure, offline functions — no PDF/network
needed. Exercises the closed Clerk bracket vocabulary against wrapped/wide PDF-layout row text
(the real failure mode measured on the corpus: the asset-type bracket or other cell text can sit
between the low and high dollar figures), the strict ``(0.0, 0.0)`` parse-miss sentinel (logged
with DocID, counted), and the open-top bracket represented distinctly from a miss.
"""

import logging

import pytest

from trading.infrastructure.data_sources.house_disclosures import (
    _OPEN_TOP_HIGH,
    _OPEN_TOP_LOW,
    _parse_amount_range,
    _parse_transactions,
)

# --- Direct _parse_amount_range fixtures: every closed bracket + open-top, wrapped layouts -----


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        # Single-line, unwrapped (the pre-existing common case) still parses for every bracket.
        ("P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]", (1_001.0, 15_000.0)),
        ("P 05/16/2025 06/09/2025 $15,001 - $50,000 [ST]", (15_001.0, 50_000.0)),
        ("P 05/16/2025 06/09/2025 $50,001 - $100,000 [ST]", (50_001.0, 100_000.0)),
        ("P 05/16/2025 06/09/2025 $100,001 - $250,000 [ST]", (100_001.0, 250_000.0)),
        ("P 05/16/2025 06/09/2025 $250,001 - $500,000 [ST]", (250_001.0, 500_000.0)),
        ("P 05/16/2025 06/09/2025 $500,001 - $1,000,000 [ST]", (500_001.0, 1_000_000.0)),
        ("P 05/16/2025 06/09/2025 $1,000,001 - $5,000,000 [ST]", (1_000_001.0, 5_000_000.0)),
        ("P 05/16/2025 06/09/2025 $5,000,001 - $25,000,000 [ST]", (5_000_001.0, 25_000_000.0)),
        ("P 05/16/2025 06/09/2025 $25,000,001 - $50,000,000 [ST]", (25_000_001.0, 50_000_000.0)),
        # Wrapped layout: the asset-type bracket (or other cell text) sits between the dash and
        # the high figure, exactly as pdfplumber renders the real Clerk PDFs (Task 07 measured
        # this on the corpus's top-3 zero-amount filers).
        (
            "SP Netflix, Inc. - Common Stock (NFLX) P 05/16/2025 06/09/2025 $15,001 -\n"
            "[ST] $50,000",
            (15_001.0, 50_000.0),
        ),
        # A row above $500,000, wrapped across a line break with intervening asset-type text.
        (
            "SP Big Corp Holdings (BIGC) P 05/16/2025 06/09/2025 $1,000,001 -\n[ST] $5,000,000",
            (1_000_001.0, 5_000_000.0),
        ),
        # Wide layout: extra whitespace/columns between the two figures on one line.
        ("P 05/16/2025 06/09/2025 $50,001    -    $100,000 [ST]", (50_001.0, 100_000.0)),
    ],
)
def test_closed_bracket_parses_including_wrapped_layouts(
    block: str, expected: tuple[float, float]
) -> None:
    assert _parse_amount_range(block) == expected


def test_open_top_bracket_parses_and_is_distinct_from_a_miss() -> None:
    """A genuine 'over $50,000,000' row must not collide with the (0.0, 0.0) miss sentinel."""
    block = "P 05/16/2025 06/09/2025 over $50,000,000 [ST]"
    low, high = _parse_amount_range(block)
    assert (low, high) == (_OPEN_TOP_LOW, _OPEN_TOP_HIGH)
    assert (low, high) != (0.0, 0.0)


def test_open_top_bracket_wrapped_across_line_break() -> None:
    block = "P 05/16/2025 06/09/2025 over\n[ST] $50,000,000"
    low, high = _parse_amount_range(block)
    assert (low, high) == (_OPEN_TOP_LOW, _OPEN_TOP_HIGH)


def test_unparseable_block_is_strict_miss_sentinel_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No bracket pattern matches at all -> (0.0, 0.0), logged with the DocID, never confused
    with the open-top bracket."""
    block = "P 05/16/2025 06/09/2025 [ST] no amount text here at all"
    with caplog.at_level(logging.INFO):
        result = _parse_amount_range(block, doc_id="99999999")
    assert result == (0.0, 0.0)
    assert any(
        "amount-bracket parse miss" in record.message and "99999999" in record.message
        for record in caplog.records
    )


def test_miss_sentinel_and_open_top_never_collide() -> None:
    assert (_OPEN_TOP_LOW, _OPEN_TOP_HIGH) != (0.0, 0.0)
    assert _OPEN_TOP_LOW > 0.0
    assert _OPEN_TOP_HIGH > 0.0


# --- _parse_transactions integration: a wrapped amount inside a real-shaped multi-line PTR row --


def test_parse_transactions_recovers_wrapped_amount_from_real_shaped_row() -> None:
    """Reproduces the real Clerk PDF-text layout (2030502.pdf's Netflix row): the ticker/date
    line ends with the low figure and a dangling dash, and the NEXT line carries the asset-type
    bracket plus the high figure."""
    text = (
        "Name: Hon. Richard W. Allen\n"
        "SP Netflix, Inc. - Common Stock (NFLX) P 05/16/2025 06/09/2025 $15,001 -\n"
        "[ST] $50,000\n"
    )
    transactions = _parse_transactions(text, doc_id="20030502")
    assert len(transactions) == 1
    txn = transactions[0]
    assert txn.ticker == "NFLX"
    assert (txn.amount_low, txn.amount_high) == (15_001.0, 50_000.0)


def test_parse_transactions_counts_and_logs_amount_miss_by_doc_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = (
        "Name: Hon. Richard W. Allen\n"
        "SP Mystery Holdings (MYST) P 05/16/2025 06/09/2025 [ST] no dollar figures here\n"
    )
    with caplog.at_level(logging.INFO):
        transactions = _parse_transactions(text, doc_id="20030999")
    assert len(transactions) == 1
    assert (transactions[0].amount_low, transactions[0].amount_high) == (0.0, 0.0)
    assert any("20030999" in record.message for record in caplog.records)
