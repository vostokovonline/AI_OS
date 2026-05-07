from canonical_skills.base import Skill, SkillResult, Artifact
from logging_config import get_logger
import yfinance as yf

logger = get_logger(__name__)

class StockAnalysisSkill(Skill):
    """Auto-generated skill for stock_analysis, market_data"""

    id = "stock_analysis_skill"
    version = "0.1.0"

    def execute(self, inputs, context):
        """Execute stock_analysis, market_data operation."""
        try:
            result_data = ""
            if 'text' in inputs and inputs['text'] == 'AAPL':
                stock = yf.Ticker("AAPL")
                hist = stock.history(period="1mo")
                analysis = f"Recent AAPL Stock Data:\n{hist.to_string()}"
                result_data = analysis

                # Return success with artifacts
                return SkillResult(
                    success=True,
                    data=result_data,
                    artifacts=[
                        Artifact(
                            type="FILE",
                            content_kind="text",
                            content_location="output.txt",
                            description="Generated output"
                        )
                    ]
                )
            else:
                result_data = "Invalid input. Please provide 'AAPL'."

                # Return success with artifacts
                return SkillResult(
                    success=True,
                    data=result_data,
                    artifacts=[
                        Artifact(
                            type="FILE",
                            content_kind="text",
                            content_location="output.txt",
                            description="Generated output"
                        )
                    ]
                )

        except Exception as e:
            logger.error("skill_execution_failed", error=str(e), exc_info=True)
            return SkillResult(
                success=False,
                error=str(e),
                artifacts=[]
            )