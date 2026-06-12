import sys
from PyQt6.QtWidgets import QApplication
from chapter_selector import ChapterSelectorForm

# Define the global modern QSS Dark Style Sheet
THEME_QSS = """
/* Dark theme for PyQt6 OpenCV Learner */
QWidget {
    background-color: #121214;
    color: #e3e3e6;
    font-family: "Malgun Gothic", "Segoe UI", sans-serif;
    font-size: 13px;
}

QTabWidget::panel {
    border: 1px solid #2d2d30;
    background-color: #18181c;
    border-radius: 6px;
    padding: 10px;
}

QTabBar::tab {
    background-color: #25252b;
    border: 1px solid #2d2d30;
    border-bottom: none;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    color: #a0a0a5;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #18181c;
    color: #ffffff;
    border-bottom: 2px solid #7f5af0;
}

QTabBar::tab:hover {
    background-color: #2e2e36;
    color: #ffffff;
}

QGroupBox {
    border: 1px solid #2d2d30;
    border-radius: 6px;
    margin-top: 15px;
    padding-top: 15px;
    font-weight: bold;
    color: #7f5af0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #121214;
}

QPushButton {
    background-color: #7f5af0;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #906cf2;
}

QPushButton:pressed {
    background-color: #6343c7;
}

QPushButton:disabled {
    background-color: #2a2a2e;
    color: #606065;
}

/* Specific button styles matching legacy C# styles */
QPushButton#btnGoToLab, QPushButton#btnRecStart, QPushButton#btnStart, QPushButton#btnStep1, QPushButton#btnStep2 {
    background-color: #2cb67d;
}
QPushButton#btnGoToLab:hover, QPushButton#btnRecStart:hover, QPushButton#btnStart:hover, QPushButton#btnStep1:hover, QPushButton#btnStep2:hover {
    background-color: #38d68f;
}
QPushButton#btnGoToLab:pressed, QPushButton#btnRecStart:pressed, QPushButton#btnStart:pressed, QPushButton#btnStep1:pressed, QPushButton#btnStep2:pressed {
    background-color: #219262;
}

QPushButton#btnStop, QPushButton#btnRecStop, QPushButton#btnPlayStop, QPushButton#btnStep3, QPushButton#btnFreeNow {
    background-color: #ef4565;
}
QPushButton#btnStop:hover, QPushButton#btnRecStop:hover, QPushButton#btnPlayStop:hover, QPushButton#btnStep3:hover, QPushButton#btnFreeNow:hover {
    background-color: #ff5675;
}
QPushButton#btnStop:pressed, QPushButton#btnRecStop:pressed, QPushButton#btnPlayStop:pressed, QPushButton#btnStep3:pressed, QPushButton#btnFreeNow:pressed {
    background-color: #d1324f;
}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1c1c1e;
    border: 1px solid #2d2d30;
    border-radius: 4px;
    padding: 6px;
    color: #e3e3e6;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QScrollBar:vertical {
    border: none;
    background: #18181c;
    width: 10px;
    margin: 0px 0 0px 0;
}
QScrollBar::handle:vertical {
    background: #3e3e42;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #4e4e52;
}

QSlider::groove:horizontal {
    border: 1px solid #2d2d30;
    height: 8px;
    background: #1e1e24;
    margin: 2px 0;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #7f5af0;
    border: none;
    width: 16px;
    height: 16px;
    margin: -4px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #906cf2;
}

QProgressBar {
    border: 1px solid #2d2d30;
    border-radius: 4px;
    text-align: center;
    background-color: #1e1e24;
}

QProgressBar::chunk {
    background-color: #2cb67d;
    border-radius: 3px;
}
"""

def main():
    app = QApplication(sys.argv)
    
    # Set the global stylesheet
    app.setStyleSheet(THEME_QSS)
    
    selector = ChapterSelectorForm()
    selector.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
