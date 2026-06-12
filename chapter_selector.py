import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QApplication
)
from PyQt6.QtGui import QFont

# Import Chapters
from ch02_image_size_fps import FormCh02
from ch03_camera_output import FormCh03
from ch04_video_playback_record import FormCh04
from ch05_image_memory_leak import FormCh05
from ch06_class_grayscale import FormCh06


class ChapterSelectorForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Python 실습 챕터 선택")
        self.resize(400, 380)
        self.setMinimumSize(380, 340)
        self.setWindowFlags(Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.CustomizeWindowHint)
        
        self.active_chapter_window = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(12)
        
        lbl_title = QLabel("학습할 실습 챕터를 선택하세요")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 15px;")
        layout.addWidget(lbl_title)
        
        # Chapter 2 Button
        btn_ch02 = QPushButton("002. Image Size & FPS 실습")
        btn_ch02.setFixedHeight(45)
        btn_ch02.clicked.connect(lambda: self.open_chapter(FormCh02))
        layout.addWidget(btn_ch02)
        
        # Chapter 3 Button
        btn_ch03 = QPushButton("003. Camera 출력 실습")
        btn_ch03.setFixedHeight(45)
        btn_ch03.clicked.connect(lambda: self.open_chapter(FormCh03))
        layout.addWidget(btn_ch03)
        
        # Chapter 4 Button
        btn_ch04 = QPushButton("004. 동영상 파일 출력 실습")
        btn_ch04.setFixedHeight(45)
        btn_ch04.clicked.connect(lambda: self.open_chapter(FormCh04))
        layout.addWidget(btn_ch04)
        
        # Chapter 5 Button
        btn_ch05 = QPushButton("005. 이미지 파일 출력 실습")
        btn_ch05.setFixedHeight(45)
        btn_ch05.clicked.connect(lambda: self.open_chapter(FormCh05))
        layout.addWidget(btn_ch05)
        
        # Chapter 6 Button
        btn_ch06 = QPushButton("006. 클래스 설계 & GrayScale 실습")
        btn_ch06.setFixedHeight(45)
        btn_ch06.clicked.connect(lambda: self.open_chapter(FormCh06))
        layout.addWidget(btn_ch06)
        
    def open_chapter(self, chapter_class):
        self.hide()
        
        # Create the chapter window
        self.active_chapter_window = chapter_class()
        
        # Override the closeEvent to show the main menu back when closed
        def on_close_callback(event):
            self.active_chapter_window = None
            self.show()
            event.accept()
            
        self.active_chapter_window.closeEvent = on_close_callback
        self.active_chapter_window.show()
