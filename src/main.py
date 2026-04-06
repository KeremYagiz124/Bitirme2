import sys
import os

# Add project root to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def create_app():
    """Create and configure the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setStyleSheet("""
    QWidget {
        background-color: #000000;
        color: #e5e7eb;
    }

    QFrame {
        background-color: #0f172a;
        border-radius: 16px;
    }

    QPushButton {
        background-color: #3b82f6;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }

    QPushButton:hover {
        background-color: #2563eb;
    }
    """)

    return app


if __name__ == "__main__":
    app = create_app()
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
