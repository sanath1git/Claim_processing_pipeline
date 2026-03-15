"""
Application configuration — loads environment variables and exposes settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the claim processing pipeline."""

    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Model selection
    SEGREGATOR_MODEL: str = "gemini-2.5-flash"       # fast model for classification
    EXTRACTION_MODEL: str = "gemini-2.5-flash"        # strong model for data extraction

    # PDF constraints
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}

    # Image rendering
    PDF_DPI: int = 200                            # resolution for page-to-image conversion
    PAGE_BATCH_SIZE: int = 4                      # pages per segregator LLM call


settings = Settings()
