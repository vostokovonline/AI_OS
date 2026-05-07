from canonical_skills.base import Skill, SkillResult, Artifact
from logging_config import get_logger
import requests

logger = get_logger(__name__)

class WeatherApiSkill(Skill):
    """Auto-generated skill for weather_api, temperature_check"""

    id = "weather_api_skill"
    version = "0.1.0"

    def execute(self, inputs, context):
        """Execute weather_api, temperature_check operation."""
        try:
            result_data = inputs.get('text', '')
            city = "London"
            api_key = "YOUR_API_KEY"  # Replace with your actual API key
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                temperature = data['main']['temp']
                result_data = f"The current temperature in {city} is {temperature}°C"
            else:
                result_data = "Failed to retrieve weather data"

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