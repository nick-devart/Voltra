"""
strategy/straddle_strategy.py
------------------------------
BTC ATM Short Straddle — NautilusTrader Strategy (Level 1 + Level 2).

NautilusTrader Strategy lifecycle — what happens and when
---------------------------------------------------------
on_start()          → Called once when the LiveNode starts all strategies.
                      We subscribe to data, set a timer for ATM selection.

on_quote_tick(qt)   → Called on every best-bid/offer update for a subscribed
                      instrument. This is where we update premium tracking and
                      evaluate SL/TP conditions.

on_order_accepted() → Called when the exchange acknowledges our order.
on_order_filled()   → Called when an order (or partial fill) completes.
on_order_rejected() → Called when an order is rejected.
on_stop()           → Called on graceful shutdown — we close the DB.

Key NautilusTrader concepts used here
--------------------------------------
- self.cache               : Access to all loaded instruments and cached ticks.
- self.order_factory       : Creates order objects (limit, market, etc.).
- self.submit_order(order) : Routes an order through the ExecutionEngine to Bybit.
- self.clock.set_timer()   : Schedule a callback at a future time.
- self.subscribe_quote_ticks(instrument_id) : Register for live bid/ask updates.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading import Strategy
from nautilus_trader.model import (
    InstrumentId,
    Quantity,
    Price,
    OrderSide,
    TimeInForce,
    QuoteTick,
    OrderFilled,
    OrderRejected,
    OrderAccepted,
    OrderCanceled,
    CryptoOption,
    StrategyId,
)

from strategy.instrument_selector import InstrumentSelector, StraddlePair
from strategy.position_manager import PositionManager, LegStatus
from strategy.risk_manager import RiskManager, ExitReason
from database.database import TradeDatabase, TradeRecord
from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class StraddleConfig(StrategyConfig):
    """
    NautilusTrader StrategyConfig subclass.

    In rc6, StrategyConfig is a plain Python class (not pydantic).
    Custom fields are declared as class-level type annotations with defaults.
    The strategy_id field expects a StrategyId object.
    """

    quantity: float = 0.1
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    max_spread_pct: float = 0.10
    max_expiry_scan: int = 5
    instrument_load_wait_secs: float = 15.0


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class StraddleStrategy(Strategy):
    """
    BTC ATM Short Straddle.

    State machine
    -------------
    WAITING_FOR_QUOTES → waiting for live quotes to populate the quote cache
    SELECTING          → running ATM instrument selection
    ENTERING           → sell orders submitted, waiting for fills
    ACTIVE             → both legs filled; monitoring premium for SL/TP
    EXITING            → close buy orders submitted
    CLOSED             → both legs closed, trade is done
    ERROR              → unrecoverable error (e.g. one leg rejected after other filled)
    """

    def __init__(self, config: StraddleConfig) -> None:
        super().__init__(config)

        # State
        self._state = "WAITING_FOR_QUOTES"
        self._pair: Optional[StraddlePair] = None
        self._trade_id: Optional[str] = None
        self._entry_time: Optional[float] = None   # unix timestamp at entry
        self._btc_ref_price: Optional[float] = None  # latest BTC reference price

        # Quote cache  instrument_id_str -> (bid, ask)
        self._quote_cache: dict[str, tuple[float, float]] = {}

        # Order IDs — tracked so we know which fills belong to which leg
        self._call_entry_oid: Optional[str] = None
        self._put_entry_oid: Optional[str] = None
        self._call_close_oid: Optional[str] = None
        self._put_close_oid: Optional[str] = None

        # Collaborators (pure Python — no NT dependency)
        self._selector = InstrumentSelector(
            max_spread_pct=config.max_spread_pct,
        )
        self._pm = PositionManager()
        self._rm = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_daily_loss_usdt=settings.safety.max_daily_loss_usdt,
        )
        self._db = TradeDatabase(settings.database.path)

        logger.info(
            f"StraddleStrategy created | qty={config.quantity} "
            f"SL={config.stop_loss_pct:.1%} TP={config.take_profit_pct:.1%}"
        )

    # ------------------------------------------------------------------
    # Lifecycle — on_start
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Called once by NautilusTrader when the LiveNode is running.

        Steps:
        1. Find all CryptoOption instruments in the cache (loaded by the
           Bybit data client when OPTION is in product_types).
        2. Subscribe to QuoteTick for every BTC option — gives us live bid/ask.
        3. Also subscribe to BTCUSDT-LINEAR for the reference price.
        4. Set a timer — after the wait period, run ATM selection.
        """
        self.log.info("on_start() — subscribing to BTC options.")

        btc_options = self._get_btc_options()
        if not btc_options:
            self.log.error(
                "No BTC CryptoOption instruments in cache. "
                "Ensure BybitProductType.OPTION is in product_types and "
                "the connection succeeded."
            )
            return

        self.log.info(f"Found {len(btc_options)} BTC option instruments — subscribing to quotes.")
        for inst in btc_options:
            self.subscribe_quotes(inst.id)

        # Subscribe to the linear perp for reference price
        try:
            linear_id = InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT")
            self.subscribe_quotes(linear_id)
        except Exception as e:
            self.log.warning(f"Could not subscribe to BTCUSDT-LINEAR: {e}")

        # Schedule ATM selection after quotes have had time to arrive
        wait = self.config.instrument_load_wait_secs
        self.log.info(f"Waiting {wait}s for initial quotes before ATM selection.")
        self.clock.set_timer(
            name="atm_select_timer",
            interval=timedelta(seconds=wait),
            callback=self._on_atm_timer,
            start_time=None,
            stop_time=None,
        )

    def _on_atm_timer(self, event) -> None:
        """
        Timer callback — fires once after the initial wait period.
        Cancel the timer then run ATM selection.
        """
        self.clock.cancel_timer("atm_select_timer")

        if self._state != "WAITING_FOR_QUOTES":
            return

        self._state = "SELECTING"
        self.log.info("Timer fired — running ATM selection.")
        self._select_and_enter()

    # ------------------------------------------------------------------
    # Instrument selection + entry
    # ------------------------------------------------------------------

    def _select_and_enter(self) -> None:
        """Run ATM selection and submit entry orders."""
        btc_options = self._get_btc_options()
        ref_price = self._get_reference_price()

        if ref_price is None:
            self.log.error("Cannot determine BTC reference price — aborting.")
            self._state = "WAITING_FOR_QUOTES"
            return

        self.log.info(f"BTC reference price: {ref_price:.2f} USDT")

        pair = self._selector.find_atm_straddle(
            instruments=btc_options,
            reference_price=ref_price,
            quotes=self._quote_cache,
            max_expiry_scan=self.config.max_expiry_scan,
        )

        if pair is None:
            self.log.error("ATM selection failed — no valid straddle pair found.")
            self._state = "WAITING_FOR_QUOTES"
            return

        self._pair = pair
        self._trade_id = str(uuid.uuid4())[:8]

        self.log.info("=" * 60)
        self.log.info("ATM STRADDLE SELECTED")
        self.log.info(pair.summary())
        self.log.info("=" * 60)

        # Initialise position manager
        self._pm.initialize(
            call_instrument_id=pair.call.id.value,
            put_instrument_id=pair.put.id.value,
            quantity=self.config.quantity,
        )

        # Persist open trade record to SQLite
        self._db.insert_trade(TradeRecord(
            trade_id=self._trade_id,
            call_instrument_id=pair.call.id.value,
            put_instrument_id=pair.put.id.value,
            strike=pair.strike,
            expiry=str(pair.expiry.date()),
            entry_time=datetime.now(timezone.utc).isoformat(),
            call_quantity=self.config.quantity,
            put_quantity=self.config.quantity,
        ))

        self._state = "ENTERING"
        self._submit_entry_orders(pair)

    def _submit_entry_orders(self, pair: StraddlePair) -> None:
        """
        Submit SELL limit orders at the current bid price for both legs.

        Why limit orders?
        We place limit SELL orders at the current bid price. This means we
        post to the order book as a maker. Options on testnet can have thin
        books — a market order could get a terrible fill or be rejected.

        GTC (Good Till Cancelled) means the order stays open until filled
        or manually cancelled.
        """
        qty = Quantity.from_str(str(self.config.quantity))

        # --- Call leg ---
        call_bid = pair.call_bid
        if not call_bid or call_bid <= 0:
            self.log.error("Call bid is unavailable — cannot submit entry orders.")
            self._state = "WAITING_FOR_QUOTES"
            return

        call_order = self.order_factory.limit(
            instrument_id=pair.call.id,
            order_side=OrderSide.SELL,
            quantity=qty,
            price=Price(call_bid, pair.call.price_precision),
            time_in_force=TimeInForce.GTC,
        )
        self._call_entry_oid = call_order.client_order_id.value
        self._pm.on_entry_order_submitted("CALL", self._call_entry_oid)
        self.submit_order(call_order)
        self.log.info(f"CALL SELL order submitted: id={self._call_entry_oid} price={call_bid}")

        # --- Put leg ---
        put_bid = pair.put_bid
        if not put_bid or put_bid <= 0:
            self.log.error("Put bid is unavailable — cannot submit entry orders.")
            self._state = "WAITING_FOR_QUOTES"
            return

        put_order = self.order_factory.limit(
            instrument_id=pair.put.id,
            order_side=OrderSide.SELL,
            quantity=qty,
            price=Price(put_bid, pair.put.price_precision),
            time_in_force=TimeInForce.GTC,
        )
        self._put_entry_oid = put_order.client_order_id.value
        self._pm.on_entry_order_submitted("PUT", self._put_entry_oid)
        self.submit_order(put_order)
        self.log.info(f"PUT  SELL order submitted: id={self._put_entry_oid} price={put_bid}")

    # ------------------------------------------------------------------
    # Event handlers — NautilusTrader calls these automatically
    # ------------------------------------------------------------------

    def on_quote(self, tick: QuoteTick) -> None:
        """
        Called on every best bid/offer update for subscribed instruments.

        1. Update our local quote cache.
        2. Keep BTC reference price fresh.
        3. Update the position manager's latest prices (for PnL and SL/TP).
        4. If ACTIVE and both legs filled → evaluate risk conditions.
        """
        iid = tick.instrument_id.value
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        self._quote_cache[iid] = (bid, ask)

        # Track live BTC reference price from BTCUSDT-LINEAR
        if "BTCUSDT-LINEAR" in iid or "BTCUSD-LINEAR" in iid:
            self._btc_ref_price = (bid + ask) / 2.0

        # Update position manager's latest market prices
        if self._pm.call and iid == self._pm.call.instrument_id:
            self._pm.on_quote_update("CALL", bid, ask)
        elif self._pm.put and iid == self._pm.put.instrument_id:
            self._pm.on_quote_update("PUT", bid, ask)

        # Risk evaluation only when both legs are live
        if self._state == "ACTIVE" and self._pm.both_legs_filled:
            signal = self._rm.evaluate(self._pm)
            if signal is not None:
                # try_mark_exit_triggered is the duplicate-exit guard
                if self._pm.try_mark_exit_triggered():
                    self.log.warning(f"Exit triggered: {signal}")
                    self._submit_close_orders(signal.reason.name)

    def on_order_accepted(self, event: OrderAccepted) -> None:
        """Exchange confirmed it received and accepted our order."""
        oid = event.client_order_id.value
        if oid == self._call_entry_oid:
            self._pm.on_entry_order_accepted("CALL")
            self.log.info("CALL entry order accepted by exchange.")
        elif oid == self._put_entry_oid:
            self._pm.on_entry_order_accepted("PUT")
            self.log.info("PUT  entry order accepted by exchange.")

    def on_order_filled(self, event: OrderFilled) -> None:
        """
        Called when any of our orders is filled (or partially filled).

        NautilusTrader calls this handler once per fill event. For partial
        fills it fires multiple times with incrementally increasing quantities.

        event.last_px  = fill price for this execution
        event.last_qty = quantity filled in this execution
        """
        oid = event.client_order_id.value
        fill_price = float(event.last_px)
        fill_qty = float(event.last_qty)

        # --- Entry fills ---
        if oid == self._call_entry_oid:
            self._pm.on_entry_fill("CALL", fill_price, fill_qty)
            self.log.info(f"CALL entry fill: {fill_qty} @ {fill_price}")

        elif oid == self._put_entry_oid:
            self._pm.on_entry_fill("PUT", fill_price, fill_qty)
            self.log.info(f"PUT  entry fill: {fill_qty} @ {fill_price}")

        # --- Close fills ---
        elif oid == self._call_close_oid:
            self._pm.on_close_fill("CALL", fill_price, fill_qty)
            self.log.info(f"CALL close fill: {fill_qty} @ {fill_price}")

        elif oid == self._put_close_oid:
            self._pm.on_close_fill("PUT", fill_price, fill_qty)
            self.log.info(f"PUT  close fill: {fill_qty} @ {fill_price}")

        # Transition: ENTERING → ACTIVE once both legs are fully filled
        if self._state == "ENTERING" and self._pm.both_legs_filled:
            self._state = "ACTIVE"
            self._entry_time = time.time()   # record when both legs were live
            entry = self._pm.combined_entry_premium
            thresholds = self._rm.thresholds(entry) if entry else {}
            self.log.info("=" * 60)
            self.log.info("BOTH LEGS FILLED — straddle is ACTIVE")
            self.log.info(f"Combined entry premium : {entry:.6f} USDT")
            if thresholds:
                self.log.info(
                    f"Stop-Loss level        : {thresholds['stop_loss_level']:.6f} USDT "
                    f"(+{thresholds['stop_loss_pct']:.1%})"
                )
                self.log.info(
                    f"Take-Profit level      : {thresholds['take_profit_level']:.6f} USDT "
                    f"(-{thresholds['take_profit_pct']:.1%})"
                )
            self.log.info("=" * 60)
            # Update DB with actual entry premiums
            if self._trade_id:
                self._db.update_entry_premiums(
                    trade_id=self._trade_id,
                    call_entry_premium=self._pm.call.entry_premium,
                    put_entry_premium=self._pm.put.entry_premium,
                    combined_entry_prem=entry,
                )

        # Transition: EXITING → CLOSED once both legs are fully closed
        if self._state == "EXITING" and self._pm.both_legs_closed:
            self._state = "CLOSED"
            self._on_straddle_closed()

    def on_order_rejected(self, event: OrderRejected) -> None:
        """
        Called when an order is rejected by the exchange.

        Feature D — partial entry handling:
        If one leg fills but the other is rejected, we have an unhedged
        position. We log a CRITICAL alert and transition to ERROR state.
        The filled leg stays open — closing it is a manual risk decision.
        """
        oid = event.client_order_id.value
        reason = str(event.reason)
        self.log.error(f"Order REJECTED: {oid} — {reason}")

        if oid == self._call_entry_oid:
            self._pm.on_entry_rejected("CALL", reason)
            self._handle_partial_rejection("CALL")
        elif oid == self._put_entry_oid:
            self._pm.on_entry_rejected("PUT", reason)
            self._handle_partial_rejection("PUT")

    def on_order_canceled(self, event: OrderCanceled) -> None:
        oid = event.client_order_id.value
        self.log.info(f"Order cancelled: {oid}")

    def _handle_partial_rejection(self, rejected_side: str) -> None:
        """Handle the case where one leg is rejected after the other has filled."""
        other = "PUT" if rejected_side == "CALL" else "CALL"
        other_leg = self._pm._leg(other)

        if other_leg.status == LegStatus.FILLED:
            self.log.critical(
                f"PARTIAL ENTRY ALERT: {rejected_side} leg REJECTED "
                f"but {other} leg is already FILLED. "
                f"You have an unhedged short {other} position on the exchange. "
                f"Manual intervention required."
            )
            self._state = "ERROR"
        else:
            self.log.warning(
                f"{rejected_side} leg rejected and {other} is not filled — "
                f"returning to WAITING_FOR_QUOTES."
            )
            self._state = "WAITING_FOR_QUOTES"

    # ------------------------------------------------------------------
    # Exit / close orders
    # ------------------------------------------------------------------

    def _submit_close_orders(self, exit_reason: str) -> None:
        """
        Submit BUY limit orders to close (buy back) both short positions.

        Uses current ask price as limit price — we are willing to pay the
        ask to exit quickly. IOC (Immediate-or-Cancel) means: fill what
        you can right now, cancel the rest. This prevents lingering orders
        that could double-fill.
        """
        if self._pair is None or self._pm.call is None or self._pm.put is None:
            return

        self._state = "EXITING"
        self.log.info(f"Submitting close orders — reason: {exit_reason}")

        # --- Close Call ---
        call_ask = self._pm.call.latest_ask
        call_qty = self._pm.call.filled_qty
        if call_ask and call_ask > 0 and call_qty > 0:
            qty = Quantity.from_str(str(call_qty))
            close_call = self.order_factory.limit(
                instrument_id=InstrumentId.from_str(self._pm.call.instrument_id),
                order_side=OrderSide.BUY,
                quantity=qty,
                price=Price(call_ask, self._pair.call.price_precision),
                time_in_force=TimeInForce.IOC,
            )
            self._call_close_oid = close_call.client_order_id.value
            self._pm.on_close_order_submitted("CALL", self._call_close_oid)
            self.submit_order(close_call)
            self.log.info(f"CALL BUY close order: id={self._call_close_oid} price={call_ask}")

        # --- Close Put ---
        put_ask = self._pm.put.latest_ask
        put_qty = self._pm.put.filled_qty
        if put_ask and put_ask > 0 and put_qty > 0:
            qty = Quantity.from_str(str(put_qty))
            close_put = self.order_factory.limit(
                instrument_id=InstrumentId.from_str(self._pm.put.instrument_id),
                order_side=OrderSide.BUY,
                quantity=qty,
                price=Price(put_ask, self._pair.put.price_precision),
                time_in_force=TimeInForce.IOC,
            )
            self._put_close_oid = close_put.client_order_id.value
            self._pm.on_close_order_submitted("PUT", self._put_close_oid)
            self.submit_order(close_put)
            self.log.info(f"PUT  BUY close order: id={self._put_close_oid} price={put_ask}")

    def _on_straddle_closed(self) -> None:
        """Both close legs are confirmed — calculate realized PnL and update DB."""
        c = self._pm.call
        p = self._pm.put

        call_entry = (c.entry_avg_price or 0.0) * c.closed_qty
        put_entry  = (p.entry_avg_price or 0.0) * p.closed_qty
        call_exit  = (c.exit_avg_price  or 0.0) * c.closed_qty
        put_exit   = (p.exit_avg_price  or 0.0) * p.closed_qty

        combined_entry = call_entry + put_entry
        combined_exit  = call_exit  + put_exit
        # Positive = profit (we collected more than we paid back)
        realized_pnl = combined_entry - combined_exit

        self.log.info("=" * 60)
        self.log.info("STRADDLE CLOSED")
        self.log.info(f"  Call: entry={c.entry_avg_price:.6f} exit={c.exit_avg_price:.6f}")
        self.log.info(f"  Put:  entry={p.entry_avg_price:.6f} exit={p.exit_avg_price:.6f}")
        self.log.info(f"  Realized PnL: {realized_pnl:+.4f} USDT")
        self.log.info("=" * 60)

        if realized_pnl < 0:
            self._rm.record_realized_loss(abs(realized_pnl))

        if self._trade_id:
            self._db.close_trade(
                trade_id=self._trade_id,
                exit_time=datetime.now(timezone.utc).isoformat(),
                call_exit_premium=call_exit,
                put_exit_premium=put_exit,
                combined_exit_prem=combined_exit,
                realized_pnl=realized_pnl,
                exit_reason="CLOSED",
            )

    # ------------------------------------------------------------------
    # Reference price helpers
    # ------------------------------------------------------------------

    def _get_btc_options(self) -> list[CryptoOption]:
        all_instruments = self.cache.instruments()
        return [
            i for i in all_instruments
            if isinstance(i, CryptoOption) and "BTC" in i.id.value
        ]

    def _get_reference_price(self) -> Optional[float]:
        """
        Get BTC spot/perp reference price.

        Priority:
        1. BTCUSDT-LINEAR quote from cache (most liquid on testnet).
        2. Any cached BTC linear quote (fallback if ID format differs).
        3. None if unavailable — selection is aborted.
        """
        try:
            linear_id = InstrumentId.from_str("BTCUSDT-LINEAR.BYBIT")
            qt = self.cache.quote(linear_id)
            if qt:
                ref = (float(qt.bid_price) + float(qt.ask_price)) / 2.0
                self.log.info(f"Reference price from BTCUSDT-LINEAR: {ref:.2f}")
                return ref
        except Exception:
            pass

        # Fallback: scan quote cache
        for iid, (bid, ask) in self._quote_cache.items():
            if "BTCUSDT-LINEAR" in iid or "BTCUSD-LINEAR" in iid:
                ref = (bid + ask) / 2.0
                self.log.info(f"Reference price from quote cache ({iid}): {ref:.2f}")
                return ref

        self.log.warning("Cannot determine BTC reference price from any source.")
        return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def on_stop(self) -> None:
        self.log.info("StraddleStrategy stopping — closing database.")
        self._db.close()

    # ------------------------------------------------------------------
    # Dashboard API (called from backend/api.py)
    # ------------------------------------------------------------------

    def get_dashboard_snapshot(self) -> dict:
        """Return current state as a JSON-serialisable dict."""
        pm_snap = self._pm.snapshot()
        entry = self._pm.combined_entry_premium
        thresholds = self._rm.thresholds(entry) if entry else {}

        # Trade duration as HH:MM:SS
        duration_str = None
        if self._entry_time is not None:
            elapsed = int(time.time() - self._entry_time)
            h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            duration_str = f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "state":                self._state,
            "trade_id":             self._trade_id,
            "btc_reference_price":  self._btc_ref_price,
            "trade_duration":       duration_str,
            "pair": {
                "strike":   self._pair.strike if self._pair else None,
                "expiry":   str(self._pair.expiry.date()) if self._pair else None,
                "call_id":  self._pair.call.id.value if self._pair else None,
                "put_id":   self._pair.put.id.value  if self._pair else None,
            },
            "positions":            pm_snap,
            "thresholds":           thresholds,
            "kill_switch_active":   self._rm.kill_switch_active,
        }

    def activate_kill_switch(self) -> None:
        """Trigger emergency exit from the dashboard."""
        self._rm.trigger_kill_switch()
        if self._state == "ACTIVE":
            if self._pm.try_mark_exit_triggered():
                self._submit_close_orders("KILL_SWITCH")
