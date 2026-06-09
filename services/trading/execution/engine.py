"""Execution engine - paper trading positions"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: Optional[PositionSide] = None
    entry_price: float = 0
    current_price: float = 0
    quantity: float = 0
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0
    pnl_pct: float = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"
    
    def unrealized_pnl(self) -> float:
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity * self.leverage
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity * self.leverage
        return 0


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: str = ""
    order_type: str = "market"
    price: float = 0
    quantity: float = 0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None


class ExecutionEngine:
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.trades_history: list[dict] = []
        logger.info("execution_engine_initialized", balance=initial_balance)
    
    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        if symbol in self.positions and self.positions[symbol].status == "open":
            logger.warning("position_already_open", symbol=symbol)
            return self.positions[symbol]
        
        cost = quantity * price / leverage
        if cost > self.balance:
            logger.error("insufficient_balance", cost=cost, balance=self.balance)
            raise ValueError(f"Insufficient balance: need {cost}, have {self.balance}")
        
        position = Position(
            symbol=symbol,
            side=PositionSide[side.upper()],
            entry_price=price,
            current_price=price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        self.positions[symbol] = position
        self.balance -= cost
        
        logger.info(
            "position_opened",
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            leverage=leverage
        )
        return position
    
    def close_position(self, symbol: str, reason: str = "signal") -> Optional[dict]:
        if symbol not in self.positions:
            logger.warning("position_not_found", symbol=symbol)
            return None
        
        position = self.positions[symbol]
        if position.status != "open":
            return None
        
        entry_cost = position.entry_price * position.quantity / position.leverage
        
        if position.side == PositionSide.LONG:
            pnl = (position.current_price - position.entry_price) * position.quantity * position.leverage
        else:
            pnl = (position.entry_price - position.current_price) * position.quantity * position.leverage
        
        self.balance += entry_cost + pnl
        position.status = "closed"
        position.pnl = pnl
        position.pnl_pct = (pnl / entry_cost) * 100
        
        trade = {
            "id": str(uuid4()),
            "symbol": symbol,
            "side": position.side.value,
            "entry_price": position.entry_price,
            "exit_price": position.current_price,
            "quantity": position.quantity,
            "leverage": position.leverage,
            "pnl": pnl,
            "pnl_pct": position.pnl_pct,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "reason": reason,
            "created_at": position.created_at.isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
        }
        
        self.trades_history.append(trade)
        logger.info("position_closed", symbol=symbol, pnl=pnl, pnl_pct=position.pnl_pct)
        
        del self.positions[symbol]
        return trade
    
    def update_price(self, symbol: str, price: float) -> Optional[dict]:
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.current_price = price
        position.updated_at = datetime.utcnow()
        position.pnl = position.unrealized_pnl()
        
        closed = None
        if position.stop_loss:
            if position.side == PositionSide.LONG and price <= position.stop_loss:
                closed = self.close_position(symbol, "stop_loss")
            elif position.side == PositionSide.SHORT and price >= position.stop_loss:
                closed = self.close_position(symbol, "stop_loss")
        
        if not closed and position.take_profit:
            if position.side == PositionSide.LONG and price >= position.take_profit:
                closed = self.close_position(symbol, "take_profit")
            elif position.side == PositionSide.SHORT and price <= position.take_profit:
                closed = self.close_position(symbol, "take_profit")
        
        return closed
    
    def get_stats(self) -> dict:
        total_pnl = sum(t["pnl"] for t in self.trades_history)
        win_trades = [t for t in self.trades_history if t["pnl"] > 0]
        loss_trades = [t for t in self.trades_history if t["pnl"] <= 0]
        
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.initial_balance) * 100,
            "open_positions": len(self.positions),
            "total_trades": len(self.trades_history),
            "win_trades": len(win_trades),
            "loss_trades": len(loss_trades),
            "win_rate": len(win_trades) / len(self.trades_history) if self.trades_history else 0,
        }
    
    def get_positions(self) -> list[dict]:
        return [
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side.value if p.side else None,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "quantity": p.quantity,
                "leverage": p.leverage,
                "unrealized_pnl": p.unrealized_pnl(),
                "pnl_pct": p.unrealized_pnl() / (p.entry_price * p.quantity / p.leverage) * 100 if p.entry_price > 0 else 0,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "created_at": p.created_at.isoformat(),
            }
            for p in self.positions.values()
        ]