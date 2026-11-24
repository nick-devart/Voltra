"""
strategy/instrument_selector.py
--------------------------------
Feature A: Smart ATM Selection

Responsible for:
  1. Filtering CryptoOption instruments to BTC-only.
  2. Grouping by (expiry, strike, option_kind).
  3. Finding the nearest expiry where both a Call and a Put exist
     at the same strike.
  4. Choosing the strike closest to the current BTC reference price.
  5. Validating that live quotes exist and that bid/ask spread is acceptable.

This module is PURE PYTHON — it does not talk to NautilusTrader or Bybit
directly. The strategy feeds it instruments and quotes from the NT cache.

CryptoOption field types (from NT docs):
  - inst.underlying    : Currency object  (e.g. Currency.from_str("BTC"))
  - inst.strike_price  : Price object     (use float(inst.strike_price))
  - inst.expiration_ns : int              (nanoseconds since epoch)
  - inst.option_kind   : OptionKind enum  (OptionKind.CALL / OptionKind.PUT)
  - inst.price_precision: int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from nautilus_trader.model import CryptoOption, OptionKind

logger = logging.getLogger(__name__)


@dataclass
class StraddlePair:
    """A validated ATM Call + Put pair ready for order submission."""
    call: CryptoOption
    put: CryptoOption
    strike: float
    expiry: datetime
    call_bid: Optional[float]
    call_ask: Optional[float]
    put_bid: Optional[float]
    put_ask: Optional[float]

    @property
    def call_mid(self) -> Optional[float]:
        if self.call_bid is not None and self.call_ask is not None:
            return (self.call_bid + self.call_ask) / 2.0
        return None

    @property
    def put_mid(self) -> Optional[float]:
        if self.put_bid is not None and self.put_ask is not None:
            return (self.put_bid + self.put_ask) / 2.0
        return None

    def summary(self) -> str:
        return (
            f"Strike={self.strike:.0f} | Expiry={self.expiry.date()} | "
            f"Call={self.call.id.value}  bid={self.call_bid} ask={self.call_ask} | "
            f"Put= {self.put.id.value}   bid={self.put_bid}  ask={self.put_ask}"
        )


class InstrumentSelector:
    """
    Selects the nearest ATM straddle pair from the NT instrument cache.

    How it works
    ------------
    When the Bybit data client connects with BybitProductType.OPTION in
    product_types, it loads ALL available option instruments into the NT
    cache as CryptoOption objects.

    We then:
    1. Filter to BTC options only (underlying.code == "BTC").
    2. Skip expired instruments.
    3. Group by expiry, then by strike, separating calls and puts.
    4. Find expiries that have BOTH a call and put at the same strike.
    5. Among those, pick the strike numerically closest to the reference price.
    6. Validate that live quotes exist and spreads are within limits.
    """

    def __init__(
        self,
        max_spread_pct: float = 0.10,
    ) -> None:
        self._max_spread_pct = max_spread_pct

    def find_atm_straddle(
        self,
        instruments: list[CryptoOption],
        reference_price: float,
        quotes: dict[str, tuple[float, float]],
        max_expiry_scan: int = 5,
    ) -> Optional[StraddlePair]:
        """
        Return the best ATM straddle pair, or None if nothing qualifies.

        Parameters
        ----------
        instruments     : CryptoOption instruments from the NT cache.
        reference_price : BTC price used to identify the ATM strike.
        quotes          : bid/ask cache keyed by instrument_id string.
        max_expiry_scan : Number of near-term expiries to try.
        """
        # Filter to BTC options — underlying is a Currency object, so use .code
        btc_options = [
            i for i in instruments
            if isinstance(i, CryptoOption) and i.underlying.code == "BTC"
        ]
        if not btc_options:
            logger.warning("No BTC CryptoOption instruments found.")
            return None

        logger.info(
            f"Scanning {len(btc_options)} BTC options, ref_price={reference_price:.2f}"
        )

        # Group by expiry datetime
        now = datetime.now(tz=timezone.utc)
        expiry_map: dict[datetime, list[CryptoOption]] = {}
        for inst in btc_options:
            exp_dt = datetime.fromtimestamp(
                inst.expiration_ns / 1e9, tz=timezone.utc
            )
            if exp_dt <= now:
                continue  # skip already-expired options
            expiry_map.setdefault(exp_dt, []).append(inst)

        if not expiry_map:
            logger.warning("All available BTC options have expired.")
            return None

        sorted_expiries = sorted(expiry_map.keys())  # nearest first

        for expiry in sorted_expiries[:max_expiry_scan]:
            pair = self._try_expiry(
                expiry=expiry,
                instruments=expiry_map[expiry],
                reference_price=reference_price,
                quotes=quotes,
            )
            if pair is not None:
                logger.info(f"Selected pair: {pair.summary()}")
                return pair

        logger.warning(
            f"No valid ATM straddle found after scanning "
            f"{min(max_expiry_scan, len(sorted_expiries))} expiries."
        )
        return None

    def _try_expiry(
        self,
        expiry: datetime,
        instruments: list[CryptoOption],
        reference_price: float,
        quotes: dict[str, tuple[float, float]],
    ) -> Optional[StraddlePair]:
        """Attempt to find a valid ATM pair for one expiry."""
        calls: dict[float, CryptoOption] = {}
        puts: dict[float, CryptoOption] = {}

        for inst in instruments:
            # strike_price is a Price object — convert to float
            strike = float(inst.strike_price)
            if inst.option_kind == OptionKind.CALL:
                calls[strike] = inst
            elif inst.option_kind == OptionKind.PUT:
                puts[strike] = inst

        # Strikes that have BOTH a call and a put
        paired = sorted(set(calls) & set(puts))
        if not paired:
            logger.debug(f"Expiry {expiry.date()}: no matched call+put pairs.")
            return None

        # Find strike closest to reference price
        nearest = min(paired, key=lambda s: abs(s - reference_price))
        logger.debug(
            f"Expiry {expiry.date()}: nearest_strike={nearest:.0f} "
            f"ref={reference_price:.2f}"
        )

        call_inst = calls[nearest]
        put_inst = puts[nearest]

        call_bid, call_ask = quotes.get(call_inst.id.value, (None, None))
        put_bid, put_ask = quotes.get(put_inst.id.value, (None, None))

        if not self._is_valid(call_inst, call_bid, call_ask, "Call"):
            return None
        if not self._is_valid(put_inst, put_bid, put_ask, "Put"):
            return None

        return StraddlePair(
            call=call_inst,
            put=put_inst,
            strike=nearest,
            expiry=expiry,
            call_bid=call_bid,
            call_ask=call_ask,
            put_bid=put_bid,
            put_ask=put_ask,
        )

    def _is_valid(
        self,
        inst: CryptoOption,
        bid: Optional[float],
        ask: Optional[float],
        label: str,
    ) -> bool:
        if bid is None or ask is None:
            logger.debug(f"{label} {inst.id.value}: no quote — skip.")
            return False
        if bid <= 0 or ask <= 0:
            logger.debug(f"{label} {inst.id.value}: zero/negative quote — skip.")
            return False
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False
        spread_pct = (ask - bid) / mid
        if spread_pct > self._max_spread_pct:
            logger.debug(
                f"{label} {inst.id.value}: spread {spread_pct:.1%} > "
                f"max {self._max_spread_pct:.1%} — skip."
            )
            return False
        return True
