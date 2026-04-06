#!/usr/bin/env python
"""
Application startup script.

Usage: python run.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    from src.main import create_app
    from src.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())