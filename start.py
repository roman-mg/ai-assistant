#!/usr/bin/env python3
"""Simple startup script for the AI Research Assistant."""

import subprocess
import sys
from pathlib import Path


def main():
    """Start the AI Research Assistant."""
    print("🤖 Starting AI Research Assistant...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if dependencies are installed
    try:
        import src
        print("✅ Dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies with: pip install -e .")
        sys.exit(1)
    
    # Start the application
    try:
        print("🚀 Starting server...")
        print("📖 API documentation will be available at: http://localhost:8000/docs")
        print("🔍 Health check: http://localhost:8000/health")
        print("💬 Chat endpoint: http://localhost:8000/chat")
        print("🌐 WebSocket: ws://localhost:8000/ws")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the main application
        subprocess.run([sys.executable, "-m", "src.main"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down AI Research Assistant...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
