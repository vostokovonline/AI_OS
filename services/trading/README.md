# Trading MVP Service

## Files Created

```
services/trading/
├── market_data/
│   ├── binance_ws.py    # WebSocket client for Binance Futures
│   └── store.py          # Candle storage + indicators (EMA, RSI, ATR)
├── strategy/
│   └── ema_crossover.py # EMA crossover + RSI filter strategy
├── execution/
│   └── engine.py        # Paper trading engine (positions, orders, PnL)
├── risk/
│   └── engine.py        # Risk management (position sizing, SL/TP, drawdown)
├── api/
│   └── endpoints.py     # FastAPI endpoints
├── service.py           # Main service orchestrator
└── README.md            # This file
```

## What's Working

1. **Market Data**: Binance WebSocket → candles → PostgreSQL
2. **Strategy**: EMA(9/21) crossover + RSI(14) filter → signals
3. **Execution**: Paper trading with positions, PnL tracking
4. **Risk**: Position sizing, stop-loss/take-profit, drawdown protection
5. **API**: FastAPI endpoints for all operations

## Next Steps

- Add Bybit WebSocket as backup
- Add more strategies (MACD, Bollinger, Volume profile)
- Add UI dashboard
- Connect to real exchange (testnet first)