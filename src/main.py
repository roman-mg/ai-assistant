import sys

import uvicorn
from loguru import logger

from src.config.settings import settings


def main():
    """Main application entry point."""
    try:
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"OpenAI model: {settings.openai_model}")
        logger.info(f"Vector store path: {settings.faiss_index_path}")

        uvicorn.run(
            "src.api.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug",
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
