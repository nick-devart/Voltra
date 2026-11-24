"""
strategy/risk_manager.py
-------------------------
Level 2: Automated Stop-Loss / Take-Profit
Feature E: Trading Safety System

The RiskManager is called by the strategy on every QuoteTick.
It evaluates exit conditions and returns an ExitSignal when triggered.

Stop-Loss  : current combined ask cost > entry premium × (1 + SL_PCT)
Take-Profit: current combined ask cost < entry premium × (1 - TP_PCT)

The combined premium uses ASK prices on both legs (buyback valuation).
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from strategy.position_manager import PositionManager

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    STOP_LOSS        = auto()
    TAKE_PROFIT      = auto()
    KILL_SWITCH      = auto()
    MAX_DAILY_LOSS   = auto()
    STALE_QUOTE      = auto()


@dataclass
class ExitSignal:
    reason: ExitReason
    combined_entry: float
    combined_current: float
    pnl_pct: float          # positive = profit, negative = loss

    def __str__(self) -> str:
        direction = "▼" if self.pnl_pct < 0 else "▲"
        return (
            f"ExitSignal({self.reason.name}) "
            f"entry={self.combined_entry:.6f} "
            f"current={self.combined_current:.6f} "
            f"pnl={direction}{abs(self.pnl_pct):.2%}"
        )


class RiskManager:
    """
    Evaluates risk conditions on every quote update.

    Design decisions
    ----------------
    - SL and TP are computed as RELATIVE changes to the combined ENTRY premium.
    - We use ASK prices for the buyback, which is the conservative (worst-case) estimate.
    - The kill switch is a simple boolean flag that any part of the system can set.
    - Daily loss tracking is accumulated across the session.

    PnL direction for a short straddle
    ------------------------------------
    We SOLD both options.  We COLLECT premium upfront.
    To exit, we must BUY BACK both options.
    
    Profit when: buyback cost < premium collected  (options decayed)
    Loss when  : buyback cost > premium collected  (options moved against us)

    SL: buyback cost increased by > 1% relative to entry  → we are losing too much
    TP: buyback cost decreased by > 2% relative to entry  → lock in the profit
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        max_daily_loss_usdt: float = 500.0,
    ) -> None:
        self._sl_pct = stop_loss_pct
        self._tp_pct = take_profit_pct
        self._max_daily_loss = max_daily_loss_usdt
        self._kill_switch: bool = False
        self._daily_realized_loss: float = 0.0
        logger.info(
            f"RiskManager: SL={stop_loss_pct:.1%} TP={take_profit_pct:.1%} "
            f"MaxDailyLoss={max_daily_loss_usdt} USDT"
        )

    # ------------------------------------------------------------------
    # Main evaluation — call this on every QuoteTick after both legs fill
    # ------------------------------------------------------------------
    def evaluate(self, pm: PositionManager) -> Optional[ExitSignal]:
        """
        Evaluate risk conditions.  Returns an ExitSignal if exit should trigger,
        or None if the position should stay open.
        """
        # Kill switch takes priority
        if self._kill_switch:
            combined_entry = pm.combined_entry_premium or 0.0
            combined_current = pm.combined_current_ask_cost or 0.0
            pnl_pct = self._pnl_pct(combined_entry, combined_current)
            signal = ExitSignal(
                reason=ExitReason.KILL_SWITCH,
                combined_entry=combined_entry,
                combined_current=combined_current,
                pnl_pct=pnl_pct,
            )
            logger.warning(f"KILL SWITCH active — {signal}")
            return signal

        combined_entry = pm.combined_entry_premium
        combined_current = pm.combined_current_ask_cost

        # Can't evaluate without both premiums
        if combined_entry is None or combined_current is None:
            return None
        if combined_entry <= 0:
            logger.warning("combined_entry_premium is zero or negative — cannot evaluate risk.")
            return None

        pnl_pct = self._pnl_pct(combined_entry, combined_current)

        # Check daily loss limit
        if self._daily_realized_loss >= self._max_daily_loss:
            signal = ExitSignal(
                reason=ExitReason.MAX_DAILY_LOSS,
                combined_entry=combined_entry,
                combined_current=combined_current,
                pnl_pct=pnl_pct,
            )
            logger.warning(f"MAX DAILY LOSS breached — {signal}")
            return signal

        # ---- STOP-LOSS --------------------------------------------------
        # current cost increased by more than SL_PCT relative to entry
        # i.e. buyback is 1%+ more expensive than when we sold
        sl_threshold = combined_entry * (1.0 + self._sl_pct)
        if combined_current >= sl_threshold:
            signal = ExitSignal(
                reason=ExitReason.STOP_LOSS,
                combined_entry=combined_entry,
                combined_current=combined_current,
                pnl_pct=pnl_pct,
            )
            logger.warning(f"STOP-LOSS triggered — {signal}")
            return signal

        # ---- TAKE-PROFIT ------------------------------------------------
        # current cost decreased by more than TP_PCT relative to entry
        # i.e. buyback is 2%+ cheaper — we lock in the profit
        tp_threshold = combined_entry * (1.0 - self._tp_pct)
        if combined_current <= tp_threshold:
            signal = ExitSignal(
                reason=ExitReason.TAKE_PROFIT,
                combined_entry=combined_entry,
                combined_current=combined_current,
                pnl_pct=pnl_pct,
            )
            logger.info(f"TAKE-PROFIT triggered — {signal}")
            return signal

        return None

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------
    def thresholds(self, combined_entry: float) -> dict:
        """Return the absolute SL/TP premium levels for display."""
        return {
            "stop_loss_level":   combined_entry * (1.0 + self._sl_pct),
            "take_profit_level": combined_entry * (1.0 - self._tp_pct),
            "stop_loss_pct":     self._sl_pct,
            "take_profit_pct":   self._tp_pct,
        }

    # ------------------------------------------------------------------
    # Kill switch (Feature E)
    # ------------------------------------------------------------------
    def trigger_kill_switch(self) -> None:
        self._kill_switch = True
        logger.critical("EMERGENCY KILL SWITCH ACTIVATED")

    def reset_kill_switch(self) -> None:
        self._kill_switch = False
        logger.info("Kill switch reset.")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    # ------------------------------------------------------------------
    # Daily loss accumulation
    # ------------------------------------------------------------------
    def record_realized_loss(self, loss_usdt: float) -> None:
        """Call with a positive number when a trade closes at a loss."""
        if loss_usdt > 0:
            self._daily_realized_loss += loss_usdt
            logger.info(
                f"Realized loss recorded: {loss_usdt:.4f} USDT "
                f"(daily total: {self._daily_realized_loss:.4f} USDT)"
            )

    def reset_daily_loss(self) -> None:
        self._daily_realized_loss = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _pnl_pct(entry: float, current: float) -> float:
        """
        Returns the PnL percentage from the SHORT seller's perspective.
        Positive = profit (current buyback cost is less than what we collected).
        Negative = loss (current buyback cost exceeds what we collected).
        """
        if entry <= 0:
            return 0.0
        # For a short straddle: profit = entry - current
        return (entry - current) / entry
