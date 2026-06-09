"""Trading service main entry point"""
import asyncio
import logging
from typing import Optional

from .market_data.binance_ws import BinanceWebSocket
from .market_data.store import CandleStore
from .strategy.ema_crossover import EMACrossoverStrategy, SignalDirection
from .execution.engine import ExecutionEngine
from .risk.engine import RiskEngine, RiskParams

logger = logging.getLogger(__name__)


class TradingService:
    def __init__(
        self,
        db_url: str,
        symbols: list[str],
        initial_balance: float = 10000
    ):
        self.candle_store = CandleStore(db_url)
        self.strategy = EMACrossoverStrategy()
        self.execution = ExecutionEngine(initial_balance)
        self.risk = RiskEngine()
        
        self.ws = BinanceWebSocket(
            symbols=symbols,
            intervals=["1m", "5m"],
            on_candle=self._on_candle,
        )
        
        self._running = False
        logger.info("trading_service_initialized", symbols=symbols)
    
    async def _on_candle(self, candle: dict) -> None:
        self.candle_store.save(candle)
        
        if not candle["is_closed"]:
            return
        
        symbol = candle["symbol"]
        
        if symbol in self.execution.positions:
            self.execution.update_price(symbol, candle["close"])
        
        candles = self.candle_store.get_recent(symbol, candle["interval"], 50)
        if not candles:
            return
        
        signal = self.strategy.analyze(candles)
        if not signal:
            return
        
        logger.info(
            "signal_generated",
            symbol=symbol,
            direction=signal.direction.value,
            confidence=signal.confidence,
            price=signal.price
        )
        
        if signal.direction in (SignalDirection.LONG, SignalDirection.SHORT):
            risk_check = self.risk.check_entry(
                signal_confidence=signal.confidence,
                current_price=signal.price,
                side=signal.direction.value,
            )
            
            if not risk_check.approved:
                logger.warning("risk_rejected", reason=risk_check.reason)
                return
            
            try:
                self.execution.open_position(
                    symbol=symbol,
                    side=signal.direction.value,
                    price=signal.price,
                    quantity=risk_check.position_size / signal.price,
                    leverage=1,
                    stop_loss=risk_check.stop_loss,
                    take_profit=risk_check.take_profit,
                )
            except Exception as e:
                logger.error("position_open_failed", error=str(e))
        
        elif signal.direction == SignalDirection.EXIT:
            trade = self.execution.close_position(symbol, reason=signal.reason)
            if trade:
                logger.info("position_closed_by_signal", **trade)
    
    async def start(self) -> None:
        self._running = True
        await self.ws.connect()
        await self.ws.listen()
    
    async def stop(self) -> None:
        self._running = False
        await self.ws.stop()
        logger.info("trading_service_stopped")
    
    def get_status(self) -> dict:
        return {
            "running": self._running,
            "balance": self.execution.balance,
            "positions": self.execution.get_positions(),
            "stats": self.execution.get_stats(),
        }