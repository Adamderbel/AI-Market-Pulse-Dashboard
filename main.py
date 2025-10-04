#!/usr/bin/env python3
"""
Main entry point for Market Dashboard application.
"""
import os
import sys
from pathlib import Path

# Add src and project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from market_dashboard.app import app, server

# This file is only used for local development
if __name__ == "__main__":
    debug = os.environ.get('DASH_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('DASH_HOST', '0.0.0.0')
    port = int(os.environ.get('DASH_PORT', '8050'))
    
    print(f"Starting Market Dashboard on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    app.run(debug=debug, host=host, port=port)
