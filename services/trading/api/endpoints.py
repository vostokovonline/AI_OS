"""FastAPI endpoints for trading"""
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..execution.engine import ExecutionEngine, PositionSide
from ..risk.engine import RiskEngine, RiskParams
from ..strategy.ema_crossover import EMACrossoverStrategy

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading API", version="1.0.0")

execution = ExecutionEngine(initial_balance=10000)
risk_engine = RiskEngine()
strategy = EMACrossoverStrategy()


class OpenPositionRequest(BaseModel):
    symbol: str
    side: str
    price: float
    quantity: float
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class ClosePositionRequest(BaseModel):
    symbol: str
    reason: str = "manual"


class RiskParamsRequest(BaseModel):
    max_position_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    min_confidence: Optional[float] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/positions")
async def get_positions():
    return {
        "positions": execution.get_positions(),
        "count": len(execution.positions),
    }


@app.get("/positions/{symbol}")
async def get_position(symbol: str):
    if symbol not in execution.positions:
        raise HTTPException(status_code=404, detail="Position not found")
    pos = execution.positions[symbol]
    return {
        "symbol": pos.symbol,
        "side": pos.side.value if pos.side else None,
        "entry_price": pos.entry_price,
        "current_price": pos.current_price,
        "quantity": pos.quantity,
        "unrealized_pnl": pos.unrealized_pnl(),
        "pnl_pct": pos.pnl_pct,
    }


@app.post("/positions/open")
async def open_position(req: OpenPositionRequest):
    try:
        position = execution.open_position(
            symbol=req.symbol,
            side=req.side,
            price=req.price,
            quantity=req.quantity,
            leverage=req.leverage,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
        )
        return {
            "success": True,
            "position_id": position.id,
            "symbol": position.symbol,
            "side": position.side.value,
        }
    except Exception as e:
        logger.error("open_position_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/positions/{symbol}/close")
async def close_position(symbol: str, req: ClosePositionRequest):
    trade = execution.close_position(symbol, reason=req.reason)
    if not trade:
        raise HTTPException(status_code=404, detail="Position not found")
    return {"success": True, "trade": trade}


@app.get("/stats")
async def get_stats():
    return execution.get_stats()


@app.get("/history")
async def get_history():
    return {"trades": execution.trades_history, "count": len(execution.trades_history)}


@app.get("/risk/params")
async def get_risk_params():
    return {
        "max_position_pct": risk_engine.params.max_position_pct,
        "max_drawdown_pct": risk_engine.params.max_drawdown_pct,
        "max_daily_loss_pct": risk_engine.params.max_daily_loss_pct,
        "min_confidence": risk_engine.params.min_confidence,
    }


@app.post("/risk/params")
async def update_risk_params(req: RiskParamsRequest):
    if req.max_position_pct is not None:
        risk_engine.params.max_position_pct = req.max_position_pct
    if req.max_drawdown_pct is not None:
        risk_engine.params.max_drawdown_pct = req.max_drawdown_pct
    if req.max_daily_loss_pct is not None:
        risk_engine.params.max_daily_loss_pct = req.max_daily_loss_pct
    if req.min_confidence is not None:
        risk_engine.params.min_confidence = req.min_confidence
    return {"success": True, "params": risk_engine.params.__dict__}


@app.post("/risk/check")
async def check_risk(confidence: float, price: float, side: str):
    result = risk_engine.check_entry(confidence, price, side)
    return {
        "approved": result.approved,
        "position_size": result.position_size,
        "stop_loss": result.stop_loss,
        "take_profit": result.take_profit,
        "reason": result.reason,
    }


@app.get("/strategy/indicators/{symbol}")
async def get_indicators(symbol: str):
    from ..market_data.store import CandleStore
    from ..market_data.store import CandleStore
    store = CandleStore("postgresql://ns_admin:password@localhost:5432/ns_core_db")
    indicators = store.get_indicators(symbol, "5m", [9, 21, 55])
    return indicators