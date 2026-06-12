import os
import sys
import time
from PyQt6.QtCore import QTimer, Qt, QElapsedTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QGroupBox, QRadioButton, QPushButton, QComboBox, 
    QSpinBox, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap
import cv2
import numpy as np

from ui_helpers import apply_size_mode, ndarray_to_qpixmap

class FormCh03(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH03 - Camera 출력 학습 및 실습")
        self.resize(1200, 740)
        self.setMinimumSize(1000, 700)
        
        # State variables
        self.capture = None
        self.frame_counter = 0
        self.fps_timer = QElapsedTimer()
        
        # Timer
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render_timer_tick)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabs = QTabWidget()
        self.tab_theory = QWidget()
        self.tab_lab = QWidget()
        
        self.tabs.addTab(self.tab_theory, "📚 1. 핵심 이론 및 자가진단 퀴즈")
        self.tabs.addTab(self.tab_lab, "🧪 2. 실습 실험실 (Interactive Lab)")
        
        main_layout.addWidget(self.tabs)
        
        self.init_theory_tab()
        self.init_lab_tab()
        
    def init_theory_tab(self):
        layout = QHBoxLayout(self.tab_theory)
        
        # Left side: Theory & Quiz
        left_layout = QVBoxLayout()
        
        lbl_title = QLabel("003. Camera 출력 및 실시간 필터 핵심 이론 학습")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f5af0; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)
        
        txt_theory = QTextEdit()
        txt_theory.setReadOnly(True)
        txt_theory.setPlainText(
            "■ [카메라 연결하기 (cv2.VideoCapture)]\n"
            "노트북이나 컴퓨터에 달린 카메라(웹캠)를 켜고 화면을 받아오기 위해 사용하는 OpenCV의 도구가 바로 `cv2.VideoCapture`입니다.\n\n"
            "★ 카메라 번호 선택:\n"
            "  - cap = cv2.VideoCapture(0) 처럼 괄호 안에 숫자를 넣습니다.\n"
            "  - 내 컴퓨터에 연결된 카메라가 여러 개일 수 있으므로 0번, 1번 등 번호를 붙여 원하는 카메라를 골라요.\n"
            "  - 해상도(화면 크기) 설정: 카메라에게 '가로 640, 세로 480 크기로 찍어줘'라고 요청할 수도 있습니다.\n\n"
            "--------------------------------------------------\n\n"
            "■ [실시간으로 사진 찰칵찰칵! (프레임 읽기)]\n"
            "카메라는 실시간으로 세상을 계속 찍고 있기 때문에, 프로그램은 타이머 주기(예: 33ms)마다 빠르게 사진을 한 장씩 빼앗아 와서 화면에 뿌려줘야 합니다.\n"
            "  - 사진 가져오기: ret, frame = cap.read()\n"
            "  - read(): 카메라가 찍은 가장 최신 사진 한 장(numpy.ndarray)을 가져오고(frame), 사진을 가져오는 데 성공했는지 여부(True/False)를 알려줍니다(ret).\n"
            "  - 이 사진을 화면에 그릴 수 있는 액자용 그림(QPixmap)으로 바꾸어서 화면(QLabel)에 띄워 줍니다.\n\n"
            "--------------------------------------------------\n\n"
            "■ [카메라 독점과 끄기 (release)]\n"
            "카메라는 학교 화장실 칸이나 코인노래방 방처럼 **'한 번에 오직 하나의 프로그램만'** 들어가서 쓸 수 있습니다.\n"
            "  - 우리 프로그램이 카메라를 켜서 사용하는 동안에는, 다른 앱(예: 줌, 디스코드, 카카오톡 페이스톡)에서는 카메라 화면이 켜지지 않고 깜깜하게 나옵니다.\n"
            "  - 만약 사용이 끝났는데 카메라를 꺼주지 않으면(release()) 다른 프로그램이 카메라를 영영 쓸 수 없게 됩니다. 그래서 일을 마치면 반드시 `cap.release()`로 전원을 꺼줘야 합니다.\n\n"
            "--------------------------------------------------\n\n"
            "■ [실시간 필터와 컴퓨터의 피로도]\n"
            "  - 흑백 필터 (cvtColor): 단순히 색상을 빼고 밝기만 표현하므로 컴퓨터가 아주 가볍게 처리해요. 속도가 느려지지 않고 부드럽습니다.\n"
            "  - 선 따기 필터 (Canny): 사진 속에서 경계선(에지)을 찾기 위해 먼지 지우기(블러), 복잡한 수학 연산(미분) 등을 매 사진마다 엄청나게 반복해야 합니다. 컴퓨터 머리(CPU)를 많이 쓰기 때문에 속도(FPS)가 느려지거나 버벅일 수 있습니다."
        )
        txt_theory.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 10px;")
        left_layout.addWidget(txt_theory)
        
        # Quiz Area
        grp_quiz = QGroupBox("✍ 자가 진단 퀴즈 (이론 검증)")
        quiz_layout = QVBoxLayout(grp_quiz)
        
        # Quiz 1
        q1_group = QGroupBox("질문 1. 웹캠 자원 정리하기")
        q1_layout = QVBoxLayout(q1_group)
        q1_lbl = QLabel("웹캠(카메라)은 한 번에 한 프로그램만 독점해서 쓰는 장치이므로, 사용 후 반드시 capture.release()를 불러 방 문을 열어(락 해제) 주어야 한다.")
        q1_lbl.setWordWrap(True)
        q1_layout.addWidget(q1_lbl)
        
        q1_opts = QHBoxLayout()
        self.rdo_q1_o = QRadioButton("O (참)")
        self.rdo_q1_x = QRadioButton("X (거짓)")
        q1_opts.addWidget(self.rdo_q1_o)
        q1_opts.addWidget(self.rdo_q1_x)
        q1_layout.addLayout(q1_opts)
        
        self.lbl_q1_result = QLabel("정답 확인 시 해설이 여기에 표시됩니다.")
        self.lbl_q1_result.setStyleSheet("color: #72727a; font-size: 11px;")
        q1_layout.addWidget(self.lbl_q1_result)
        quiz_layout.addWidget(q1_group)
        
        # Quiz 2
        q2_group = QGroupBox("질문 2. 화면 필터 연산과 부하")
        q2_layout = QVBoxLayout(q2_group)
        q2_lbl = QLabel("실시간 카메라 화면에 선을 따주는 'Canny 필터' 같은 계산이 무거운 필터를 매 사진마다 적용하더라도, 컴퓨터 속도(FPS)는 전혀 느려지지 않는다.")
        q2_lbl.setWordWrap(True)
        q2_layout.addWidget(q2_lbl)
        
        q2_opts = QHBoxLayout()
        self.rdo_q2_o = QRadioButton("O (참)")
        self.rdo_q2_x = QRadioButton("X (거짓)")
        q2_opts.addWidget(self.rdo_q2_o)
        q2_opts.addWidget(self.rdo_q2_x)
        q2_layout.addLayout(q2_opts)
        
        self.lbl_q2_result = QLabel("정답 확인 시 해설이 여기에 표시됩니다.")
        self.lbl_q2_result.setStyleSheet("color: #72727a; font-size: 11px;")
        q2_layout.addWidget(self.lbl_q2_result)
        quiz_layout.addWidget(q2_group)
        
        btn_check = QPushButton("정답 확인 및 해설 보기")
        btn_check.setObjectName("btnCheckAnswers")
        btn_check.clicked.connect(self.check_answers)
        quiz_layout.addWidget(btn_check)
        
        btn_goto_lab = QPushButton("실습 실험실로 이동하기 (Go to Lab) ▶")
        btn_goto_lab.setObjectName("btnGoToLab")
        btn_goto_lab.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        quiz_layout.addWidget(btn_goto_lab)
        
        left_layout.addWidget(grp_quiz)
        layout.addLayout(left_layout, 1)
        
        # Right side: Diagram
        right_layout = QVBoxLayout()
        lbl_diag_title = QLabel("🖼 시각 자료: 카메라 프레임 획득 & 필터 파이프라인")
        lbl_diag_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f5af0;")
        right_layout.addWidget(lbl_diag_title)
        
        self.pic_diagram = QLabel()
        self.pic_diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_diagram.setStyleSheet("background-color: #101012; border: 1px solid #2d2d30; border-radius: 6px;")
        
        # Load ch03_camera_pipeline_ko.png
        img_path = self.get_resource_path("ch03_camera_pipeline_ko.png")
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.pic_diagram.setPixmap(pix.scaled(580, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pic_diagram.setText("[이미지 파일 없음: ch03_camera_pipeline_ko.png]")
            
        right_layout.addWidget(self.pic_diagram)
        
        txt_diag_desc = QTextEdit()
        txt_diag_desc.setReadOnly(True)
        txt_diag_desc.setPlainText(
            "【인포그래픽 설명】\n"
            "1. 웹캠 장치: 아날로그 빛 신호를 디지털 전기 신호로 실시간 캡처하여 스트림으로 방출합니다.\n"
            "2. 비디오 캡처 객체 (cv2.VideoCapture): 시스템 드라이버로부터 카메라 프레임을 초고속 획득하여 Python 메모리 영역에 numpy.ndarray 형식으로 적재합니다.\n"
            "3. 필터 연산 (cv2.cvtColor / cv2.Canny): 매 프레임 Tick 루프 돌 때마다 흑백(Grayscale) 변환 및 캐니(Canny) 에지 검출 연산이 NumPy 배열 위에서 고속 수행되어 이미지의 형태를 변화시킵니다.\n"
            "4. 화면 출력 (PyQt6 QPixmap): 처리된 NumPy 배열을 PyQt6가 이해할 수 있도록 QImage와 QPixmap으로 고속 변환하여 QLabel UI에 실시간 갱신합니다."
        )
        txt_diag_desc.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 5px;")
        txt_diag_desc.setMaximumHeight(150)
        right_layout.addWidget(txt_diag_desc)
        
        layout.addLayout(right_layout, 1)

    def init_lab_tab(self):
        layout = QHBoxLayout(self.tab_lab)
        
        # Left side: PictureBox (QLabel)
        left_layout = QVBoxLayout()
        self.pic_box = QLabel()
        self.pic_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_box.setStyleSheet("background-color: #000000; border: 1px solid #2d2d30; border-radius: 6px;")
        self.pic_box.setFixedSize(720, 480)
        left_layout.addWidget(self.pic_box)
        
        txt_guide = QTextEdit()
        txt_guide.setReadOnly(True)
        txt_guide.setPlainText("실습 가이드: PC에 연결된 웹캠 번호를 입력한 뒤 [장치 켜기]를 누르세요. 흑백 및 에지 필터 적용 시의 FPS 차이와 실제 획득되는 해상도를 학습해 보세요.")
        txt_guide.setStyleSheet("background-color: #2b2b1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        txt_guide.setMaximumHeight(50)
        left_layout.addWidget(txt_guide)
        layout.addLayout(left_layout)
        
        # Right side: Controls
        panel = QGroupBox("카메라 실습 설정")
        panel_layout = QVBoxLayout(panel)
        
        # 1. Camera Index
        panel_layout.addWidget(QLabel("1. 카메라 장치 번호 (Index)"))
        self.nud_camera_index = QSpinBox()
        self.nud_camera_index.setRange(0, 5)
        self.nud_camera_index.setValue(0)
        panel_layout.addWidget(self.nud_camera_index)
        
        # 2. Camera Resolution
        panel_layout.addWidget(QLabel("2. 카메라 목표 해상도"))
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems([
            "Default (기본 해상도)",
            "320 x 240",
            "640 x 480",
            "1280 x 720"
        ])
        self.cmb_resolution.setCurrentIndex(0)
        panel_layout.addWidget(self.cmb_resolution)
        
        # 3. Real-time Filter
        panel_layout.addWidget(QLabel("3. 실시간 처리 필터"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems([
            "None (원본 영상)",
            "Grayscale (흑백 변환)",
            "Canny Edge (경계선 감지)"
        ])
        self.cmb_filter.setCurrentIndex(0)
        panel_layout.addWidget(self.cmb_filter)
        
        # 4. Frame Interval
        panel_layout.addWidget(QLabel("4. 프레임 주기 (Interval, ms)"))
        self.nud_interval = QSpinBox()
        self.nud_interval.setRange(10, 100)
        self.nud_interval.setValue(33)
        self.nud_interval.valueChanged.connect(self.interval_changed)
        panel_layout.addWidget(self.nud_interval)
        
        # 5. SizeMode
        panel_layout.addWidget(QLabel("5. PictureBox 출력 모드"))
        self.cmb_sizemode = QComboBox()
        self.cmb_sizemode.addItems([
            "Normal (좌상단 정렬)",
            "StretchImage (비율무시 채움)",
            "Zoom (비율유지 채움)",
            "CenterImage (중앙 배치)"
        ])
        self.cmb_sizemode.setCurrentIndex(2)
        panel_layout.addWidget(self.cmb_sizemode)
        
        panel_layout.addWidget(QLabel("■ 카메라 상태 및 성능"))
        self.lbl_resolution_info = QLabel("출력 해상도: -")
        self.lbl_resolution_info.setStyleSheet("font-weight: bold; color: #2cb67d;")
        panel_layout.addWidget(self.lbl_resolution_info)
        
        self.lbl_measured_fps = QLabel("측정 FPS: 0 (정지됨)")
        self.lbl_measured_fps.setStyleSheet("font-weight: bold; color: #ef4565;")
        panel_layout.addWidget(self.lbl_measured_fps)
        
        self.lbl_status = QLabel("상태: 연결 대기 중")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(self.lbl_status)
        
        # Start/Stop Buttons
        btn_box = QHBoxLayout()
        self.btn_start = QPushButton("장치 켜기")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start_camera)
        
        self.btn_stop = QPushButton("장치 끄기")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_camera)
        
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_stop)
        panel_layout.addLayout(btn_box)
        
        # Principle text
        txt_principle = QTextEdit()
        txt_principle.setReadOnly(True)
        txt_principle.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 카메라 연결: CvCapture.FromCamera(idx) [cv2.VideoCapture(idx)]\n"
            "   - 하드웨어 웹캠으로부터 비디오 입력을 초기화합니다.\n"
            "2. 해상도 제어: SetCaptureProperty(..) [cap.set(..)]\n"
            "   - FrameWidth, FrameHeight 속성을 지정합니다.\n"
            "3. 리소스 해제: Dispose() 필수!\n"
            "   - 사용이 끝난 웹캠은 반납하지 않으면 락(Lock)이 걸려 다른 앱에서 사용 불가능합니다."
        )
        txt_principle.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 4px; padding: 5px;")
        panel_layout.addWidget(txt_principle)
        
        layout.addWidget(panel)

    def get_resource_path(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "resources", filename)
        if os.path.exists(path):
            return path
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path
        return None

    def check_answers(self):
        # Q1 Check (Answer: O)
        if self.rdo_q1_o.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q1_result.setText("정답! 카메라 장치 방 문을 열어주지 않고 종료하면, 다른 프로그램이 카메라를 켜지 못하고 에러가 나게 돼요.")
        elif self.rdo_q1_x.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q1_result.setText("오답입니다. 하드웨어 장치를 다 사용한 뒤에는 반드시 release()를 불러서 꺼줘야 락이 걸리지 않아요.")
        else:
            self.lbl_q1_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q1_result.setText("답안을 먼저 체크해 주세요.")
            
        # Q2 Check (Answer: X)
        if self.rdo_q2_x.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q2_result.setText("정답! 컴퓨터가 매 사진마다 선을 따기 위해 힘든 계산을 하느라 머리(CPU)를 많이 쓰면 화면이 뚝뚝 끊기게 돼요.")
        elif self.rdo_q2_o.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q2_result.setText("오답입니다. 계산량이 많은 무거운 작업을 추가하면 화면 재생 속도(FPS)가 느려지는 게 자연스러운 과학 원리랍니다.")
        else:
            self.lbl_q2_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q2_result.setText("답안을 먼저 체크해 주세요.")

    def interval_changed(self):
        if self.render_timer.isActive():
            self.render_timer.setInterval(self.nud_interval.value())

    def start_camera(self):
        try:
            self.stop_camera()
            
            idx = self.nud_camera_index.value()
            self.capture = cv2.VideoCapture(idx)
            
            if not self.capture.isOpened():
                QMessageBox.warning(self, "장치 연결 실패", "카메라 장치를 열 수 없습니다.")
                self.lbl_status.setText("상태: 카메라 연결 실패")
                self.capture = None
                return
                
            # Set target resolution if not default
            res_idx = self.cmb_resolution.currentIndex()
            if res_idx > 0:
                if res_idx == 1:
                    w, h = 320, 240
                elif res_idx == 3:
                    w, h = 1280, 720
                else:
                    w, h = 640, 480
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
            # Try to grab a frame to verify
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.capture.release()
                self.capture = None
                QMessageBox.warning(self, "프레임 획득 실패", "카메라로부터 프레임을 가져올 수 없습니다. 권한 혹은 연결 상태를 확인하세요.")
                self.lbl_status.setText("상태: 프레임 획득 실패")
                return
                
            actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.lbl_resolution_info.setText(f"실제 획득 해상도: {actual_w} x {actual_h}")
            
            self.frame_counter = 0
            self.fps_timer.start()
            self.render_timer.start(self.nud_interval.value())
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.nud_camera_index.setEnabled(False)
            self.cmb_resolution.setEnabled(False)
            self.lbl_status.setText("상태: 카메라 캡처 중")
            
        except Exception as ex:
            QMessageBox.critical(self, "카메라 초기화 오류", f"에러: {str(ex)}")
            self.lbl_status.setText("상태: 장치 시작 에러")
            self.stop_camera()

    def stop_camera(self):
        self.render_timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
        self.pic_box.clear()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.nud_camera_index.setEnabled(True)
        self.cmb_resolution.setEnabled(True)
        self.lbl_status.setText("상태: 연결 종료")
        self.lbl_measured_fps.setText("측정 FPS: 0 (정지됨)")

    def render_timer_tick(self):
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return
            
        # Process filter
        filter_idx = self.cmb_filter.currentIndex()
        if filter_idx == 1: # Grayscale
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_idx == 2: # Canny
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.Canny(gray, 50, 150)
        else: # None
            processed = frame
            
        # Apply size mode and display
        disp = apply_size_mode(processed, 720, 480, self.cmb_sizemode.currentText())
        pix = ndarray_to_qpixmap(disp)
        self.pic_box.setPixmap(pix)
        
        # Calculate measured FPS
        self.frame_counter += 1
        elapsed = self.fps_timer.elapsed()
        if elapsed >= 1000:
            measured_fps = self.frame_counter / (elapsed / 1000.0)
            self.lbl_measured_fps.setText(f"측정 FPS: {measured_fps:.1f}")
            self.frame_counter = 0
            self.fps_timer.restart()

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
