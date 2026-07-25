"""Transaction-side (P/S/E) parsing for the House Clerk PTR adapter (review F11).

``_transaction_code``/``_transaction_sentiment``/``_parse_transactions`` are pure, offline — no
PDF/network. These fixtures are derived from the real Clerk PDF-text layouts (see the committed
``fixtures/house/*.pdf`` rows: owner code, asset name, ``(TICKER)``, the standalone P/S/E code
column, then two dates and the amount). The bug they pin: the old first-standalone-``[PSE]``-
anywhere search matched INSIDE asset names — "S&P" (standalone S) flipped a purchase to a sale,
"J.P." (standalone P) flipped a sale to a purchase, a "(S)" ticker inverted the side, and a stray
"E" (e.g. "Series E") silently dropped a real P/S row as an exchange. The code is now read
POSITIONALLY: the standalone code between the ticker parenthetical and the first date.
"""

import pytest

from trading.infrastructure.data_sources.house_disclosures import (
    _BUY_SENTIMENT,
    _SELL_SENTIMENT,
    _parse_transactions,
    _transaction_code,
    _transaction_sentiment,
)


@pytest.mark.parametrize(
    ("block", "expected_code"),
    [
        # "S&P …" PURCHASE: the standalone S inside "S&P" must NOT win — the code column P does.
        ("SP S&P 500 Index Fund (SPY) P 05/16/2025 06/09/2025 $15,001 - $50,000 [ST]", "P"),
        # "J.P. Morgan…" SALE: the standalone P inside "J.P." must NOT win — the code column S does.
        ("JT J.P. Morgan Chase & Co (JPM) S 05/16/2025 06/09/2025 $50,001 - $100,000 [ST]", "S"),
        # single-letter ticker "(S)" must NOT invert a purchase.
        ("SP Sprint Corp (S) P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]", "P"),
        # a stray standalone "E" in the asset name must NOT read as an exchange over a real P.
        ("SP Series E Preferred Stock (XYZ) P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]", "P"),
        # a GENUINE exchange in the code column is still read as E (skipped downstream).
        ("SP Some Balanced Fund (ABC) E 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]", "E"),
        # ticker rendered AFTER the date column (the real GSK single-txn fixture layout): the code
        # before the date still wins, unaffected by there being no ticker parenthetical ahead of it.
        (
            "GSK plc American Depositary Shares S 07/28/2025 08/11/2025 "
            "$1,001 - $15,000 (GSK) [ST]",
            "S",
        ),
        # no ticker parenthetical AND a stray "E" in the asset name: the standalone code nearest the
        # date (the last one before it) wins.
        ("SP Series E Preferred Stock P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]", "P"),
    ],
)
def test_transaction_code_is_read_positionally(block: str, expected_code: str) -> None:
    assert _transaction_code(block) == expected_code


def test_sp_fund_purchase_reads_buy_not_sale() -> None:
    block = "SP S&P 500 Index Fund (SPY) P 05/16/2025 06/09/2025 $15,001 - $50,000 [ST]"
    assert _transaction_sentiment(block) == _BUY_SENTIMENT
    assert _transaction_sentiment(block) > 0  # BUY, not the mis-parsed SALE


def test_jp_morgan_sale_reads_sell_not_purchase() -> None:
    block = "JT J.P. Morgan Chase & Co (JPM) S 05/16/2025 06/09/2025 $50,001 - $100,000 [ST]"
    assert _transaction_sentiment(block) == _SELL_SENTIMENT
    assert _transaction_sentiment(block) < 0  # SELL, not the mis-parsed PURCHASE


def test_single_letter_ticker_does_not_invert_side() -> None:
    block = "SP Sprint Corp (S) P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]"
    assert _transaction_sentiment(block) > 0  # a "(S)" ticker no longer flips a purchase to a sale


def test_stray_e_in_asset_name_does_not_silently_drop_a_real_purchase() -> None:
    """A real P/S row whose asset name contains a standalone letter (here "Series E") is EMITTED
    with the correct side, not silently dropped as an exchange (review F11)."""
    text = (
        "Name: Hon. Test Member\n"
        "SP Series E Preferred Stock (XYZ) P 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]\n"
    )
    transactions = _parse_transactions(text, doc_id="20031111")
    assert len(transactions) == 1
    assert transactions[0].ticker == "XYZ"
    assert transactions[0].sentiment == _BUY_SENTIMENT


def test_genuine_exchange_row_is_skipped_not_misread() -> None:
    """When E genuinely IS the code column, the row is skipped (exchange has no directional read) —
    correctly classified, not a mis-parse."""
    text = (
        "Name: Hon. Test Member\n"
        "SP Some Balanced Fund (ABC) E 05/16/2025 06/09/2025 $1,001 - $15,000 [ST]\n"
    )
    assert _parse_transactions(text, doc_id="20032222") == []
