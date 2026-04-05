import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow

if __name__ == "__main__":
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

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
