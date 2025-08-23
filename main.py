#!/usr/bin/env python3
"""
Main entry point for Market Dashboard application.
"""
import sys
from pathlib import Path

# Add src and project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from market_dashboard.app import run_app

if __name__ == "__main__":
    run_app()
