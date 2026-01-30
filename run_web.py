#!/usr/bin/env python3
"""
Run Julie Web Interface

Start the FastAPI server for the web interface.
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Run the web server."""
    print("🚀 Starting Julie Web Interface...")
    print("=" * 50)
    print()
    print("📍 Home:   http://localhost:8000/")
    print("🎤 Voice:  http://localhost:8000/voice")
    print("⚙️  Admin:  http://localhost:8000/admin")
    print("📚 API:    http://localhost:8000/docs")
    print()
    print("=" * 50)
    print()
    
    uvicorn.run(
        "julie.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
