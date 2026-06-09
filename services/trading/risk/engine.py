"""Risk management engine"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskParams:
    max_position_pct: float = 0.02
    max_drawdown_pct: float = 0.05
    max_daily_loss_pct: float = 0.02
    min_confidence: float = 0.6
    max_leverage: int = 10
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04


@dataclass
class RiskCheckResult:
    approved: bool
    position_size: float = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class RiskEngine:
    def __init__(self, params: Optional[RiskParams] = None):
        self.params = params or RiskParams()
        self.daily_loss = 0
        self.peak_equity = 0
        self.current_equity = 0
    
    def check_entry(
        self,
        signal_confidence: float,
        current_price: float,
        side: str,
    ) -> RiskCheckResult:
        if signal_confidence < self.params.min_confidence:
            return RiskCheckResult(
                approved=False,
                reason=f"Confidence {signal_confidence:.2f} below minimum {self.params.min_confidence}"
            )
        
        if self.daily_loss >= self.params.max_daily_loss_pct * self.current_equity:
            return RiskCheckResult(
                approved=False,
                reason="Daily loss limit reached"
            )
        
        position_size = self.current_equity * self.params.max_position_pct
        
        if side == "long":
            stop_loss = current_price * (1 - self.params.default_stop_loss_pct)
            take_profit = current_price * (1 + self.params.default_take_profit_pct)
        else:
            stop_loss = current_price * (1 + self.params.default_stop_loss_pct)
            take_profit = current_price * (1 - self.params.default_take_profit_pct)
        
        return RiskCheckResult(
            approved=True,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"Position size: {position_size:.2f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}"
        )
    
    def update_equity(self, equity: float, daily_loss: float) -> None:
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.daily_loss = daily_loss
    
    def check_drawdown(self) -> bool:
        if self.peak_equity == 0:
            return True
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return drawdown < self.params.max_drawdown_pct