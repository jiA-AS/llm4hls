import openai
import logging
from .base import ModelBackend, ALPACA_PROMPT

logger = logging.getLogger(__name__)


class DeepSeekAPIBackend(ModelBackend):
    """DeepSeek API backend for HLS code generation."""
    
    def __init__(self, api_key: str, model: str = "deepseek-coder"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.max_new_tokens = 1024
        self.temperature = 0.8
        logger.info(f"DeepSeek API backend initialized with model: {model}")

    def generate(self, instruction: str) -> str:
        """Generate HLS code using DeepSeek API."""
        try:
            # Format instruction with Alpaca prompt
            prompt = ALPACA_PROMPT.format(instruction=instruction)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            
            result = response.choices[0].message.content
            logger.info("API call successful")
            return result
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return ""

    def close(self) -> None:
        """Cleanup - not needed for API backend."""
        logger.info("DeepSeek API backend closed")