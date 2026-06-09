"""Binance WebSocket Market Data Client"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any

import websockets
from websockets.client import connect as ws_connect

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """WebSocket client for Binance Futures market data"""
    
    STREAM_URL = "wss://fstream.binance.com:9443/ws"
    
    def __init__(
        self,
        symbols: list[str],
        on_candle: Optional[Callable] = None,
        on_ticker: Optional[Callable] = None,
        intervals: list[str] = None
    ):
        self.symbols = [s.lower() for s in symbols]
        self.intervals = intervals or ["1m", "5m", "15m", "1h"]
        self.on_candle = on_candle
        self.on_ticker = on_ticker
        self._running = False
        self._ws = None
        self._reconnect_delay = 1
        self._max_reconnect_delay = 30
        
    def _build_stream_url(self) -> str:
        streams = []
        for symbol in self.symbols:
            for interval in self.intervals:
                streams.append(f"{symbol}@kline_{interval}")
            streams.append(f"{symbol}@ticker")
        return f"{self.STREAM_URL}/{'/'.join(streams)}"
    
    async def connect(self) -> None:
        url = self._build_stream_url()
        logger.info("ws_connecting", url=url[:100])
        self._ws = await ws_connect(url, ping_interval=20)
        self._running = True
        self._reconnect_delay = 1
        logger.info("ws_connected", symbols=self.symbols)
    
    async def listen(self) -> None:
        while self._running:
            try:
                async for msg in self._ws:
                    data = json.loads(msg)
                    await self._handle_message(data)
            except websockets.ConnectionClosed:
                logger.warning("ws_disconnected", reconnecting=True)
                await self._reconnect()
            except Exception as e:
                logger.error("ws_error", error=str(e))
                await self._reconnect()
    
    async def _reconnect(self) -> None:
        self._running = False
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
        try:
            await self.connect()
        except Exception:
            logger.error("ws_reconnect_failed")
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        event_type = data.get("e", "")
        
        if event_type == "kline":
            await self._handle_candle(data)
        elif event_type == "24hrMiniTicker":
            await self._handle_ticker(data)
    
    async def _handle_candle(self, data: Dict[str, Any]) -> None:
        kline = data.get("k", {})
        candle = {
            "symbol": data["s"],
            "interval": kline["i"],
            "open_time": datetime.fromtimestamp(kline["t"] / 1000),
            "close_time": datetime.fromtimestamp(kline["T"] / 1000),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline["q"]),
            "is_closed": kline["x"],
            "trades": kline["n"],
        }
        if self.on_candle:
            await self.on_candle(candle)
    
    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        ticker = {
            "symbol": data["s"],
            "price_change": float(data["p"]),
            "price_change_pct": float(data["P"]),
            "last_price": float(data["c"]),
            "high": float(data["h"]),
            "low": float(data["l"]),
            "volume": float(data["v"]),
            "quote_volume": float(data["q"]),
        }
        if self.on_ticker:
            await self.on_ticker(ticker)
    
    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("ws_stopped")