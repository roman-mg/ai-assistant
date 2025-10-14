#!/usr/bin/env python3
"""Simple startup script for the AI Research Assistant."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from loguru import logger


def main() -> None:
    """Start the AI Research Assistant."""
    logger.info("🤖 Starting AI Research Assistant...")
    logger.info("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        logger.info("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)

    # Check if dependencies are installed
    try:
        importlib.util.find_spec("src")

        logger.info("✅ Dependencies are installed")
    except ImportError as e:
        logger.info(f"❌ Missing dependencies: {e}")
        logger.info("Please install dependencies with: pip install -e .")
        sys.exit(1)

    # Start the application
    try:
        logger.info("🚀 Starting server...")
        logger.info("📖 API documentation will be available at: http://localhost:8000/docs")
        logger.info("🔍 Health check: http://localhost:8000/health")
        logger.info("💬 Chat endpoint: http://localhost:8000/chat")
        logger.info("🌐 WebSocket: ws://localhost:8000/ws")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 50 + "\n")

        # Run the main application
        subprocess.run([sys.executable, "-m", "src.main"], check=True)

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down AI Research Assistant...")
    except subprocess.CalledProcessError as e:
        logger.info(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        logger.info(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
