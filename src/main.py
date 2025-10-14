import sys
import traceback

import uvicorn
from loguru import logger

from src.config.settings import settings


def main() -> None:
    """Main application entry point."""
    try:
        logger.info(f"Starting {settings.app.name} v{settings.app.version}")
        logger.info(f"Debug mode: {settings.app.debug}")

        uvicorn.run(
            "src.api.main:app",
            host=settings.server.host,
            port=settings.server.port,
            reload=settings.app.debug,
            log_level="info" if not settings.app.debug else "debug",
        )

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception:
        logger.error(f"Error starting application: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
