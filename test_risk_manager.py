"""
tests/test_risk_manager.py
--------------------------
Unit tests for the RiskManager — no exchange connection required.
"""

import pytest
from strategy.risk_manager import RiskManager, ExitReason
from strategy.position_manager import PositionManager, LegStatus


def _make_active_pm(
    call_entry: float,
    put_entry: float,
    call_ask: float,
    put_ask: float,
    qty: float = 1.0,
) -> PositionManager:
    """Helper: build a PositionManager in a fully-filled, active state."""
    pm = PositionManager()
    pm.initialize("BTC-CALL.BYBIT", "BTC-PUT.BYBIT", quantity=qty)

    pm.on_entry_order_submitted("CALL", "call-oid-1")
    pm.on_entry_order_accepted("CALL")
    pm.on_entry_fill("CALL", call_entry, qty)

    pm.on_entry_order_submitted("PUT", "put-oid-1")
    pm.on_entry_order_accepted("PUT")
    pm.on_entry_fill("PUT", put_entry, qty)

    pm.on_quote_update("CALL", call_ask * 0.9, call_ask)
    pm.on_quote_update("PUT",  put_ask  * 0.9, put_ask)

    return pm


class TestStopLoss:
    def test_sl_not_triggered_below_threshold(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        # entry = 1.0 total, current = 1.005 (0.5% increase) — below 1%
        pm = _make_active_pm(0.5, 0.5, 0.503, 0.502)
        assert rm.evaluate(pm) is None

    def test_sl_triggered_at_threshold(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        # entry = 1.0, current = 1.01 (exactly 1% increase)
        pm = _make_active_pm(0.5, 0.5, 0.505, 0.505)
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.reason == ExitReason.STOP_LOSS

    def test_sl_triggered_above_threshold(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        # entry = 1.0, current = 1.05 (5% loss)
        pm = _make_active_pm(0.5, 0.5, 0.525, 0.525)
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.reason == ExitReason.STOP_LOSS

    def test_sl_pnl_pct_is_negative_on_loss(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        pm = _make_active_pm(0.5, 0.5, 0.52, 0.52)
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.pnl_pct < 0


class TestTakeProfit:
    def test_tp_not_triggered_above_threshold(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        # entry = 1.0, current = 0.985 (1.5% decrease) — not enough
        pm = _make_active_pm(0.5, 0.5, 0.493, 0.492)
        assert rm.evaluate(pm) is None

    def test_tp_triggered_at_threshold(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        # entry = 1.0, current = 0.98 (exactly 2% decrease)
        pm = _make_active_pm(0.5, 0.5, 0.49, 0.49)
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.reason == ExitReason.TAKE_PROFIT

    def test_tp_pnl_pct_is_positive_on_profit(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        pm = _make_active_pm(0.5, 0.5, 0.48, 0.48)
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.pnl_pct > 0


class TestKillSwitch:
    def test_kill_switch_overrides_no_signal(self):
        rm = RiskManager(stop_loss_pct=0.01, take_profit_pct=0.02)
        pm = _make_active_pm(0.5, 0.5, 0.501, 0.501)  # no natural trigger
        assert rm.evaluate(pm) is None
        rm.trigger_kill_switch()
        signal = rm.evaluate(pm)
        assert signal is not None
        assert signal.reason == ExitReason.KILL_SWITCH


class TestCombinedPremiumCalculation:
    def test_combined_entry_premium_uses_both_legs(self):
        pm = _make_active_pm(0.4, 0.3, 0.41, 0.31, qty=2.0)
        # call: 0.4 * 2 = 0.8, put: 0.3 * 2 = 0.6, combined = 1.4
        assert abs(pm.combined_entry_premium - 1.4) < 1e-9

    def test_combined_ask_cost_uses_ask_prices(self):
        pm = _make_active_pm(0.4, 0.3, 0.42, 0.32, qty=2.0)
        # call ask cost: 0.42 * 2 = 0.84
        # put  ask cost: 0.32 * 2 = 0.64
        expected = 0.84 + 0.64
        assert abs(pm.combined_current_ask_cost - expected) < 1e-9

    def test_returns_none_before_fills(self):
        pm = PositionManager()
        pm.initialize("C.BYBIT", "P.BYBIT", 1.0)
        assert pm.combined_entry_premium is None
        assert pm.combined_current_ask_cost is None


class TestDuplicateExitPrevention:
    def test_first_exit_allowed(self):
        pm = PositionManager()
        pm.initialize("C.BYBIT", "P.BYBIT", 1.0)
        assert pm.try_mark_exit_triggered() is True

    def test_duplicate_exit_blocked(self):
        pm = PositionManager()
        pm.initialize("C.BYBIT", "P.BYBIT", 1.0)
        pm.try_mark_exit_triggered()
        assert pm.try_mark_exit_triggered() is False
