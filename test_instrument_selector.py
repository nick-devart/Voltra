"""
tests/test_instrument_selector.py
----------------------------------
Unit tests for InstrumentSelector — no exchange connection required.

We build real CryptoOption objects to test the selection logic so that
isinstance() checks in the selector pass correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from strategy.instrument_selector import InstrumentSelector
from nautilus_trader.model import (
    CryptoOption,
    Currency,
    InstrumentId,
    OptionKind,
    Price,
    Quantity,
)


def _make_crypto_option(
    symbol: str,
    strike: float,
    option_kind: OptionKind,
    expiry_dt: datetime,
) -> CryptoOption:
    """
    Build a real CryptoOption for use in tests.
    Uses BTC as underlying, USDT as quote/settlement.
    """
    btc = Currency.from_str("BTC")
    usdt = Currency.from_str("USDT")
    iid = InstrumentId.from_str(f"{symbol}.BYBIT")
    return CryptoOption(
        instrument_id=iid,
        raw_symbol=iid.symbol,
        underlying=btc,
        quote_currency=usdt,
        settlement_currency=usdt,
        is_inverse=False,
        option_kind=option_kind,
        strike_price=Price(strike, 1),
        activation_ns=0,
        expiration_ns=int(expiry_dt.timestamp() * 1_000_000_000),
        price_precision=2,
        size_precision=1,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.1"),
        ts_event=0,
        ts_init=0,
    )


class TestATMSelection:
    def setup_method(self):
        self.selector = InstrumentSelector(max_spread_pct=0.15)
        self.expiry = datetime.now(timezone.utc) + timedelta(days=7)

    def _make_instruments(self, strikes: list[float]) -> list:
        insts = []
        for s in strikes:
            insts.append(_make_crypto_option(f"BTC-{int(s)}-C", s, OptionKind.CALL, self.expiry))
            insts.append(_make_crypto_option(f"BTC-{int(s)}-P", s, OptionKind.PUT,  self.expiry))
        return insts

    def _make_quotes(self, strikes: list[float]) -> dict:
        quotes = {}
        for s in strikes:
            mid = max(50.0, 300.0)
            bid = mid * 0.95
            ask = mid * 1.05
            quotes[f"BTC-{int(s)}-C.BYBIT"] = (bid, ask)
            quotes[f"BTC-{int(s)}-P.BYBIT"] = (bid, ask)
        return quotes

    def test_selects_nearest_strike(self):
        strikes = [90000.0, 95000.0, 100000.0, 105000.0]
        insts = self._make_instruments(strikes)
        quotes = self._make_quotes(strikes)
        pair = self.selector.find_atm_straddle(insts, 97000.0, quotes)
        assert pair is not None
        assert pair.strike == 95000.0

    def test_selects_closest_above(self):
        strikes = [90000.0, 95000.0, 100000.0]
        insts = self._make_instruments(strikes)
        quotes = self._make_quotes(strikes)
        # 97500 is equidistant between 95000 and 100000; min() picks first
        pair = self.selector.find_atm_straddle(insts, 102000.0, quotes)
        assert pair is not None
        assert pair.strike == 100000.0

    def test_returns_none_when_no_instruments(self):
        pair = self.selector.find_atm_straddle([], 95000.0, {})
        assert pair is None

    def test_returns_none_when_no_quotes(self):
        strikes = [95000.0]
        insts = self._make_instruments(strikes)
        pair = self.selector.find_atm_straddle(insts, 95000.0, {})
        assert pair is None

    def test_rejects_wide_spread(self):
        """Spread exceeding max_spread_pct should disqualify the contract."""
        selector = InstrumentSelector(max_spread_pct=0.05)
        strikes = [95000.0]
        insts = self._make_instruments(strikes)
        # bid=1 ask=100 → spread >> 5%
        quotes = {
            "BTC-95000-C.BYBIT": (1.0, 100.0),
            "BTC-95000-P.BYBIT": (1.0, 100.0),
        }
        pair = selector.find_atm_straddle(insts, 95000.0, quotes)
        assert pair is None

    def test_expired_options_skipped(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        insts = [
            _make_crypto_option("BTC-95000-C", 95000.0, OptionKind.CALL, past),
            _make_crypto_option("BTC-95000-P", 95000.0, OptionKind.PUT,  past),
        ]
        quotes = {
            "BTC-95000-C.BYBIT": (100.0, 105.0),
            "BTC-95000-P.BYBIT": (100.0, 105.0),
        }
        pair = self.selector.find_atm_straddle(insts, 95000.0, quotes)
        assert pair is None

    def test_pair_contains_both_sides(self):
        strikes = [95000.0]
        insts = self._make_instruments(strikes)
        quotes = self._make_quotes(strikes)
        pair = self.selector.find_atm_straddle(insts, 95000.0, quotes)
        assert pair is not None
        assert pair.call.option_kind == OptionKind.CALL
        assert pair.put.option_kind == OptionKind.PUT

    def test_non_btc_options_ignored(self):
        """ETH options in the same list must not be selected."""
        future = datetime.now(timezone.utc) + timedelta(days=7)
        eth = Currency.from_str("ETH")
        usdt = Currency.from_str("USDT")

        def _eth_option(sym: str, strike: float, kind: OptionKind) -> CryptoOption:
            iid = InstrumentId.from_str(f"{sym}.BYBIT")
            return CryptoOption(
                instrument_id=iid,
                raw_symbol=iid.symbol,
                underlying=eth,
                quote_currency=usdt,
                settlement_currency=usdt,
                is_inverse=False,
                option_kind=kind,
                strike_price=Price(strike, 1),
                activation_ns=0,
                expiration_ns=int(future.timestamp() * 1_000_000_000),
                price_precision=2,
                size_precision=1,
                price_increment=Price.from_str("0.01"),
                size_increment=Quantity.from_str("0.1"),
                ts_event=0,
                ts_init=0,
            )

        insts = [
            _eth_option("ETH-3000-C", 3000.0, OptionKind.CALL),
            _eth_option("ETH-3000-P", 3000.0, OptionKind.PUT),
        ]
        quotes = {
            "ETH-3000-C.BYBIT": (100.0, 105.0),
            "ETH-3000-P.BYBIT": (100.0, 105.0),
        }
        pair = self.selector.find_atm_straddle(insts, 3000.0, quotes)
        assert pair is None
