"""Simple EMA Crossover + RSI strategy"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"
    NEUTRAL = "neutral"


@dataclass
class StrategySignal:
    direction: SignalDirection
    symbol: str
    confidence: float  # 0-1
    price: float
    reason: str
    indicators: dict


class EMACrossoverStrategy:
    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def analyze(self, candles: list[dict]) -> Optional[StrategySignal]:
        if len(candles) < self.slow_period + 5:
            return None
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        
        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)
        rsi = self._rsi(closes, self.rsi_period)
        atr = self._atr(highs, lows, closes, 14)
        
        indicators = {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "rsi": rsi,
            "atr": atr,
            "crossover": "bullish" if fast_ema > slow_ema else "bearish",
        }
        
        current_price = closes[-1]
        reason = ""
        direction = SignalDirection.NEUTRAL
        confidence = 0.5
        
        prev_fast = self._ema(closes[:-1], self.fast_period)
        prev_slow = self._ema(closes[:-1], self.slow_period)
        
        if prev_fast <= slow_ema and fast_ema > slow_ema:
            if rsi < self.rsi_oversold:
                direction = SignalDirection.LONG
                confidence = 0.85
                reason = f"Bullish crossover + RSI oversold ({rsi:.1f})"
            elif rsi < 50:
                direction = SignalDirection.LONG
                confidence = 0.7
                reason = f"Bullish crossover + RSI neutral ({rsi:.1f})"
        
        elif prev_fast >= slow_ema and fast_ema < slow_ema:
            if rsi > self.rsi_overbought:
                direction = SignalDirection.SHORT
                confidence = 0.85
                reason = f"Bearish crossover + RSI overbought ({rsi:.1f})"
            elif rsi > 50:
                direction = SignalDirection.SHORT
                confidence = 0.7
                reason = f"Bearish crossover + RSI neutral ({rsi:.1f})"
        
        elif abs(fast_ema - slow_ema) / slow_ema < 0.001:
            if rsi > self.rsi_overbought:
                direction = SignalDirection.EXIT
                confidence = 0.6
                reason = f"Neutral zone + RSI overbought ({rsi:.1f})"
            elif rsi < self.rsi_oversold:
                direction = SignalDirection.EXIT
                confidence = 0.6
                reason = f"Neutral zone + RSI oversold ({rsi:.1f})"
        
        return StrategySignal(
            direction=direction,
            symbol=candles[-1]["symbol"],
            confidence=confidence,
            price=current_price,
            reason=reason,
            indicators=indicators,
        )
    
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