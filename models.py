"""
AI model initialization and configuration
"""
import time
from google import genai
from config import Config
from utils.logger import logger
from google.genai.types import GenerateContentConfig, SafetySetting, HarmCategory, HarmBlockThreshold

def init_gemini_model():
    """Initialize Gemini model with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=Config.GEMINI_API_KEY)
            
            generation_config = GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=2048
            )
            
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            ]
            
            logger.info("✅ Gemini AI configured successfully")
            return client, generation_config, safety_settings
            
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

CLIENT, GENERATION_CONFIG, SAFETY_SETTINGS = init_gemini_model()
