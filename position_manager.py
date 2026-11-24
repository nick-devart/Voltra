"""
strategy/position_manager.py
-----------------------------
Feature C: Position and PnL Monitor

Tracks the two legs of the straddle independently and calculates
combined premium and PnL.  This is a pure in-memory data structure
— it does not call NautilusTrader directly.

The strategy feeds fill events into this manager and reads from it
when making risk decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


class LegStatus(Enum):
    PENDING   = auto()   # Order submitted, not yet acknowledged
    OPEN      = auto()   # Order acknowledged, waiting for fill
    FILLED    = auto()   # Fully filled, position is live
    PARTIAL   = auto()   # Partially filled
    CLOSING   = auto()   # Close order submitted
    CLOSED    = auto()   # Position fully closed
    CANCELLED = auto()   # Order cancelled / rejected


@dataclass
class LegState:
    """State for a single option leg (call OR put)."""
    instrument_id: str
    side: str                       # "CALL" or "PUT"

    # Entry
    entry_order_id: Optional[str] = None
    entry_avg_price: Optional[float] = None
    filled_qty: float = 0.0
    target_qty: float = 0.0
    status: LegStatus = LegStatus.PENDING

    # Exit
    close_order_id: Optional[str] = None
    exit_avg_price: Optional[float] = None
    closed_qty: float = 0.0

    # Latest market data
    latest_bid: Optional[float] = None
    latest_ask: Optional[float] = None

    @property
    def entry_premium(self) -> Optional[float]:
        """Total premium collected = fill price × quantity (for a SELL)."""
        if self.entry_avg_price is not None and self.filled_qty > 0:
            return self.entry_avg_price * self.filled_qty
        return None

    @property
    def current_ask_cost(self) -> Optional[float]:
        """Current cost to buy back = ask × filled qty."""
        if self.latest_ask is not None and self.filled_qty > 0:
            return self.latest_ask * self.filled_qty
        return None

    @property
    def unrealized_pnl(self) -> Optional[float]:
        """
        For a short option:
          PnL = premium collected - current buyback cost
        Positive means we are winning (option decayed).
        """
        ep = self.entry_premium
        cost = self.current_ask_cost
        if ep is not None and cost is not None:
            return ep - cost
        return None


class PositionManager:
    """
    Maintains state for both legs of the straddle.

    Usage flow
    ----------
    1. on_entry_order_submitted(leg, order_id, qty)
    2. on_entry_order_accepted(leg, order_id)
    3. on_entry_fill(leg, fill_price, fill_qty)      ← may be called multiple times
    4. on_quote_update(leg, bid, ask)                ← called on every QuoteTick
    5. on_close_order_submitted(leg, order_id)
    6. on_close_fill(leg, fill_price, fill_qty)
    """

    def __init__(self) -> None:
        self.call: Optional[LegState] = None
        self.put: Optional[LegState] = None
        self._exit_triggered: bool = False   # duplicate-exit guard

    def initialize(
        self,
        call_instrument_id: str,
        put_instrument_id: str,
        quantity: float,
    ) -> None:
        """Call once after the straddle pair is selected."""
        self.call = LegState(
            instrument_id=call_instrument_id,
            side="CALL",
            target_qty=quantity,
        )
        self.put = LegState(
            instrument_id=put_instrument_id,
            side="PUT",
            target_qty=quantity,
        )
        self._exit_triggered = False
        logger.info(
            f"PositionManager initialised | "
            f"Call={call_instrument_id} | Put={put_instrument_id} | qty={quantity}"
        )

    # ------------------------------------------------------------------
    # Entry lifecycle
    # ------------------------------------------------------------------
    def on_entry_order_submitted(self, side: str, order_id: str) -> None:
        leg = self._leg(side)
        leg.entry_order_id = order_id
        leg.status = LegStatus.PENDING
        logger.info(f"{side} entry order submitted: {order_id}")

    def on_entry_order_accepted(self, side: str) -> None:
        leg = self._leg(side)
        leg.status = LegStatus.OPEN
        logger.info(f"{side} entry order accepted.")

    def on_entry_fill(self, side: str, fill_price: float, fill_qty: float) -> None:
        leg = self._leg(side)
        # Running average price
        total_qty = leg.filled_qty + fill_qty
        if total_qty > 0:
            prev_cost = (leg.entry_avg_price or 0.0) * leg.filled_qty
            leg.entry_avg_price = (prev_cost + fill_price * fill_qty) / total_qty
        leg.filled_qty = total_qty

        if leg.filled_qty >= leg.target_qty:
            leg.status = LegStatus.FILLED
        else:
            leg.status = LegStatus.PARTIAL

        logger.info(
            f"{side} fill: price={fill_price} qty={fill_qty} "
            f"total_filled={leg.filled_qty}/{leg.target_qty} "
            f"avg_price={leg.entry_avg_price:.6f}"
        )

    def on_entry_rejected(self, side: str, reason: str) -> None:
        leg = self._leg(side)
        leg.status = LegStatus.CANCELLED
        logger.warning(f"{side} entry order REJECTED: {reason}")

    # ------------------------------------------------------------------
    # Quote updates — called on every QuoteTick
    # ------------------------------------------------------------------
    def on_quote_update(self, side: str, bid: float, ask: float) -> None:
        leg = self._leg(side)
        leg.latest_bid = bid
        leg.latest_ask = ask

    # ------------------------------------------------------------------
    # Exit lifecycle
    # ------------------------------------------------------------------
    def try_mark_exit_triggered(self) -> bool:
        """
        Returns True the FIRST time it is called after a fresh entry.
        Subsequent calls return False — this is the duplicate-exit guard.
        """
        if self._exit_triggered:
            logger.warning("Exit already triggered — ignoring duplicate request.")
            return False
        self._exit_triggered = True
        return True

    def on_close_order_submitted(self, side: str, order_id: str) -> None:
        leg = self._leg(side)
        leg.close_order_id = order_id
        leg.status = LegStatus.CLOSING
        logger.info(f"{side} close order submitted: {order_id}")

    def on_close_fill(self, side: str, fill_price: float, fill_qty: float) -> None:
        leg = self._leg(side)
        total = leg.closed_qty + fill_qty
        prev_cost = (leg.exit_avg_price or 0.0) * leg.closed_qty
        if total > 0:
            leg.exit_avg_price = (prev_cost + fill_price * fill_qty) / total
        leg.closed_qty = total
        if leg.closed_qty >= leg.filled_qty:
            leg.status = LegStatus.CLOSED
        logger.info(
            f"{side} close fill: price={fill_price} qty={fill_qty} "
            f"closed={leg.closed_qty}/{leg.filled_qty}"
        )

    # ------------------------------------------------------------------
    # Combined metrics — used by the risk manager
    # ------------------------------------------------------------------
    @property
    def combined_entry_premium(self) -> Optional[float]:
        if self.call is None or self.put is None:
            return None
        ce = self.call.entry_premium
        pe = self.put.entry_premium
        if ce is not None and pe is not None:
            return ce + pe
        return None

    @property
    def combined_current_ask_cost(self) -> Optional[float]:
        """Cost to buy back both legs at current ask prices."""
        if self.call is None or self.put is None:
            return None
        cc = self.call.current_ask_cost
        pc = self.put.current_ask_cost
        if cc is not None and pc is not None:
            return cc + pc
        return None

    @property
    def both_legs_filled(self) -> bool:
        if self.call is None or self.put is None:
            return False
        return (
            self.call.status == LegStatus.FILLED
            and self.put.status == LegStatus.FILLED
        )

    @property
    def both_legs_closed(self) -> bool:
        if self.call is None or self.put is None:
            return False
        return (
            self.call.status == LegStatus.CLOSED
            and self.put.status == LegStatus.CLOSED
        )

    def snapshot(self) -> dict:
        """Return a JSON-serialisable summary for the dashboard."""
        def _leg_dict(leg: Optional[LegState]) -> dict:
            if leg is None:
                return {}
            return {
                "instrument_id":    leg.instrument_id,
                "status":           leg.status.name,
                "filled_qty":       leg.filled_qty,
                "target_qty":       leg.target_qty,
                "entry_avg_price":  leg.entry_avg_price,
                "latest_bid":       leg.latest_bid,
                "latest_ask":       leg.latest_ask,
                "entry_premium":    leg.entry_premium,
                "current_ask_cost": leg.current_ask_cost,
                "unrealized_pnl":   leg.unrealized_pnl,
            }

        return {
            "call":                     _leg_dict(self.call),
            "put":                      _leg_dict(self.put),
            "combined_entry_premium":   self.combined_entry_premium,
            "combined_current_ask_cost":self.combined_current_ask_cost,
            "exit_triggered":           self._exit_triggered,
            "both_legs_filled":         self.both_legs_filled,
            "both_legs_closed":         self.both_legs_closed,
        }

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _leg(self, side: str) -> LegState:
        if side.upper() == "CALL":
            if self.call is None:
                raise RuntimeError("PositionManager: call leg not initialized.")
            return self.call
        elif side.upper() == "PUT":
            if self.put is None:
                raise RuntimeError("PositionManager: put leg not initialized.")
            return self.put
        else:
            raise ValueError(f"Unknown leg side: {side}")
