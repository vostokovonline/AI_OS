from canonical_skills.base import Skill, SkillResult, Artifact
from logging_config import get_logger
import requests
import json

logger = get_logger(__name__)

class StockFetcherSkill(Skill):
    """Auto-generated skill for stock_fetcher"""

    id = "stock_fetcher_skill"
    version = "0.1.0"

    def execute(self, inputs, context):
        """Execute stock_fetcher operation."""
        try:
            result_data = {}
            stock_symbol = inputs.get('text', '').strip().upper()

            if stock_symbol == 'TSLA':
                url = f'https://api.iextrading.com/1.0/stock/{stock_symbol}/quote'
                response = requests.get(url)
                if response.status_code == 200:
                    result_data = response.json()
                else:
                    raise Exception(f"Failed to fetch stock data: {response.status_code}")
            else:
                raise Exception("Invalid stock symbol")

            # Return success with artifacts
            return SkillResult(
                success=True,
                data=json.dumps(result_data),
                artifacts=[
                    Artifact(
                        type="FILE",
                        content_kind="json",
                        content_location="output.json",
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