"""Candle storage with PostgreSQL"""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Index, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


class Candle(Base):
    __tablename__ = "candles"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(10), nullable=False)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, nullable=False)
    trades = Column(Integer, default=0)
    is_closed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_candle_symbol_interval_time", "symbol", "interval", "open_time"),
    )


class CandleStore:
    def __init__(self, connection_url: str):
        self.engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
        )
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("candle_store_initialized")
    
    def save(self, candle: dict) -> None:
        session = self.Session()
        try:
            db_candle = Candle(
                symbol=candle["symbol"],
                interval=candle["interval"],
                open_time=candle["open_time"],
                close_time=candle["close_time"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
                quote_volume=candle["quote_volume"],
                trades=candle.get("trades", 0),
                is_closed=candle.get("is_closed", False),
            )
            session.add(db_candle)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("candle_save_failed", symbol=candle["symbol"], error=str(e))
            raise
        finally:
            session.close()
    
    def get_recent(self, symbol: str, interval: str, limit: int = 100) -> list[dict]:
        session = self.Session()
        try:
            candles = (
                session.query(Candle)
                .filter(Candle.symbol == symbol, Candle.interval == interval)
                .order_by(Candle.open_time.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "symbol": c.symbol,
                    "interval": c.interval,
                    "open_time": c.open_time,
                    "close_time": c.close_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "quote_volume": c.quote_volume,
                    "trades": c.trades,
                    "is_closed": c.is_closed,
                }
                for c in reversed(candles)
            ]
        finally:
            session.close()
    
    def get_indicators(self, symbol: str, interval: str, periods: list[int]) -> dict:
        candles = self.get_recent(symbol, interval, max(max(periods) + 10, 200))
        if len(candles) < max(periods) + 1:
            return {}
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        
        result = {}
        
        for period in periods:
            result[f"ema_{period}"] = self._ema(closes, period)
            result[f"rsi_{period}"] = self._rsi(closes, period)
        
        result["atr_14"] = self._atr(highs, lows, closes, 14)
        
        return result
    
    def _ema(self, prices: list[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _rsi(self, prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = sum(gains) / period or 1
        avg_loss = sum(losses) / period or 1
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
        if len(closes) < period:
            return 0
        trs = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1]) if i > -len(closes) else 0
            low_close = abs(lows[i] - closes[i-1]) if i > -len(closes) else 0
            trs.append(max(high_low, high_close, low_close))
        return sum(trs) / period if trs else 0