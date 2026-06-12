import os
import sys
import time
from PyQt6.QtCore import QTimer, Qt, QElapsedTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QGroupBox, QRadioButton, QPushButton, QComboBox, 
    QSpinBox, QSplitter, QStyle
)
from PyQt6.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np

from ui_helpers import apply_size_mode, ndarray_to_qpixmap

class FormCh02(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH02 - Image Size & FPS 학습 및 실습")
        self.resize(1200, 740)
        self.setMinimumSize(1000, 700)
        
        # State variables
        self.ball_x = 100.0
        self.ball_y = 100.0
        self.ball_speed_x = 6.0
        self.ball_speed_y = 4.0
        self.ball_radius = 20
        self.frame_counter = 0
        self.fps_timer = QElapsedTimer()
        
        # Timer
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render_timer_tick)
        
        self.init_ui()
        self.update_calculations()
        
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
        
        lbl_title = QLabel("002. Image Size & FPS 핵심 이론 학습")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f5af0; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)
        
        txt_theory = QTextEdit()
        txt_theory.setReadOnly(True)
        txt_theory.setPlainText(
            "■ [컴퓨터가 이미지를 그리는 모눈종이 판]\n"
            "컴퓨터 화면 속 사진이나 그림은 아주 미세한 색상 점들이 가로, 세로 격자 판 위에 빽빽하게 깔려서 만들어집니다.\n"
            "이 작은 색상 점 하나를 '픽셀(Pixel)'이라고 불러요.\n\n"
            "★ 이미지의 메모리 크기 계산 방법:\n"
            "이미지 용량(바이트) = 가로 크기(점 개수) × 세로 크기(점 개수) × 채널 수(종이 겹 수)\n"
            "  - 그레이스케일(흑백): 1채널. 검은색부터 흰색까지의 밝기만 표현하므로, 점 하나당 1바이트 크기면 충분해요.\n"
            "  - 컬러(BGR): 3채널. 빛의 3원색인 파랑(Blue), 초록(Green), 빨강(Red) 색종이를 3장 겹쳐서 풍부한 색을 만듭니다. 그래서 점 하나당 3바이트 크기를 써요.\n"
            "  - 꿀팁: 컬러 이미지는 흑백 이미지보다 용량이 정확히 3배 더 무겁답니다!\n\n"
            "--------------------------------------------------\n\n"
            "■ [움직이는 동영상의 마법: FPS와 대기 시간]\n"
            "동영상은 교과서 모서리에 그림을 한 장씩 그려두고 손가락으로 주르륵 넘기는 '플립북' 놀이와 똑같아요.\n"
            "정지된 사진들을 엄청 빠르게 번갈아 보여주면 눈이 착각을 일으켜 움직이는 것처럼 보이는 것이죠.\n"
            "  - FPS (Frames Per Second): 1초 동안 눈앞에 지나가는 사진 장수예요. 숫자가 클수록 움직임이 아주 부드러워져요.\n"
            "  - 대기 시간 (Interval, ms): 사진 한 장을 띄우고 다음 장으로 넘어가기 전까지 멈춰 있는 아주 짧은 대기 시간(1000분의 1초 단위)이에요.\n"
            "  - 공식: 목표 FPS = 1000 / 대기 시간(ms)\n"
            "    - 대기 시간이 33ms이면: 1000 / 33 ≒ 30 FPS (TV나 유튜브 영상처럼 자연스러워요)\n"
            "    - 대기 시간이 16ms이면: 1000 / 16 ≒ 60 FPS (게임 화면처럼 엄청 매끄러워요)\n\n"
            "--------------------------------------------------\n\n"
            "■ [화면 크기 맞추기 모드 (SizeMode)]\n"
            "불러온 사진의 크기와 화면 액자의 크기가 다를 때 어떻게 채울지 결정하는 네 가지 방식이에요.\n"
            "  - Normal (그대로 보기): 원본 크기 그대로 왼쪽 위부터 채웁니다. 사진이 액자보다 크면 삐져나온 부분은 잘려요.\n"
            "  - StretchImage (찌그러트려 꽉 채우기): 사진 비율이 망가지거나 찌그러지더라도 액자 크기에 맞게 억지로 늘려서 가득 채워요.\n"
            "  - Zoom (비율 유지하며 채우기): 사진 비율을 예쁘게 유지하면서 액자 안에 쏙 들어가게 조절해요. 남는 부분은 검은색 테두리(여백)로 남겨둡니다.\n"
            "  - CenterImage (가운데 정렬): 사진을 줄이거나 늘리지 않고, 딱 한가운데를 기준으로 화면에 맞춰 보여줍니다."
        )
        txt_theory.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 10px;")
        left_layout.addWidget(txt_theory)
        
        # Quiz Area
        grp_quiz = QGroupBox("✍ 자가 진단 퀴즈 (이론 검증)")
        quiz_layout = QVBoxLayout(grp_quiz)
        
        # Quiz 1
        q1_group = QGroupBox("질문 1. BGR 이미지 메모리 크기 계산")
        q1_layout = QVBoxLayout(q1_group)
        q1_lbl = QLabel("가로 1280칸, 세로 720칸의 모눈종이에 3장씩 겹쳐진(3채널 BGR 컬러) 이미지가 차지하는 용량은 대략 2,764,800 바이트(약 2.64MB)이다.")
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
        q2_group = QGroupBox("질문 2. 화면 맞춤 모드(SizeMode) 구분")
        q2_layout = QVBoxLayout(q2_group)
        q2_lbl = QLabel("StretchImage 모드는 이미지 고유의 비율을 유지하지 않고, 화면 크기에 맞추기 위해 가로세로를 강제로 늘려 꽉 채운다.")
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
        lbl_diag_title = QLabel("🖼 시각 자료: 이미지 구조 및 픽셀 매핑")
        lbl_diag_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f5af0;")
        right_layout.addWidget(lbl_diag_title)
        
        self.pic_diagram = QLabel()
        self.pic_diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_diagram.setStyleSheet("background-color: #101012; border: 1px solid #2d2d30; border-radius: 6px;")
        
        # Load ch02_pixel_structure_ko.png
        img_path = self.get_resource_path("ch02_pixel_structure_ko.png")
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.pic_diagram.setPixmap(pix.scaled(580, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pic_diagram.setText("[이미지 파일 없음: ch02_pixel_structure_ko.png]")
            
        right_layout.addWidget(self.pic_diagram)
        
        txt_diag_desc = QTextEdit()
        txt_diag_desc.setReadOnly(True)
        txt_diag_desc.setPlainText(
            "【인포그래픽 설명】\n"
            "1. 왼쪽(단일 채널 - 그레이스케일): 픽셀 하나당 1바이트(8비트, 0~255) 데이터만을 사용합니다. 명암 강도만을 표현하므로 메모리를 최소한으로 소모합니다.\n"
            "2. 오른쪽(3채널 - BGR 컬러): 픽셀 하나를 구성하기 위해 Blue(청색), Green(녹색), Red(적색) 각각 1바이트씩 총 3바이트(24비트)의 메모리를 소모합니다. 픽셀 당 3개 성분의 조합을 통해 다채로운 색을 구현합니다.\n"
            "3. 결론: 컬러 이미지는 동일 해상도의 흑백 이미지보다 데이터양이 정확히 3배가 되어 하드웨어 대역폭과 계산 성능에 상당한 부담을 줍니다. 따라서 고속 처리가 생명인 컴퓨터 비전 파이프라인에서는 컬러 이미지를 그레이스케일로 변환하는 작업을 전처리 1순위로 수행합니다."
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
        txt_guide.setPlainText("실습 가이드: 각 옵션을 설정한 뒤 [시작] 버튼을 눌러 공의 움직임과 FPS 변동을 관찰해 보세요.")
        txt_guide.setStyleSheet("background-color: #2b2b1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        txt_guide.setMaximumHeight(50)
        left_layout.addWidget(txt_guide)
        layout.addLayout(left_layout)
        
        # Right side: Controls
        panel = QGroupBox("실습 제어 및 모니터링")
        panel_layout = QVBoxLayout(panel)
        
        # 1. Resolution
        panel_layout.addWidget(QLabel("1. 이미지 해상도 선택 (Width x Height)"))
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems([
            "320 x 240 (QVGA)", 
            "640 x 480 (VGA)", 
            "1280 x 720 (HD)", 
            "1920 x 1080 (FHD)"
        ])
        self.cmb_resolution.setCurrentIndex(1)
        self.cmb_resolution.currentIndexChanged.connect(self.update_calculations)
        panel_layout.addWidget(self.cmb_resolution)
        
        # 2. Channels
        panel_layout.addWidget(QLabel("2. 컬러 채널 수 (Color Channels)"))
        self.cmb_channels = QComboBox()
        self.cmb_channels.addItems([
            "1 Channel (Grayscale)",
            "3 Channels (BGR Color)"
        ])
        self.cmb_channels.setCurrentIndex(1)
        self.cmb_channels.currentIndexChanged.connect(self.update_calculations)
        panel_layout.addWidget(self.cmb_channels)
        
        # 3. Interval
        panel_layout.addWidget(QLabel("3. 프레임 갱신 주기 (Interval, ms)"))
        self.nud_interval = QSpinBox()
        self.nud_interval.setRange(5, 1000)
        self.nud_interval.setValue(33)
        self.nud_interval.valueChanged.connect(self.interval_changed)
        panel_layout.addWidget(self.nud_interval)
        
        # 4. SizeMode
        panel_layout.addWidget(QLabel("4. PictureBox 출력 모드 (SizeMode)"))
        self.cmb_sizemode = QComboBox()
        self.cmb_sizemode.addItems([
            "Normal (좌상단 정렬)",
            "StretchImage (비율무시 채움)",
            "Zoom (비율유지 채움)",
            "CenterImage (중앙 배치)"
        ])
        self.cmb_sizemode.setCurrentIndex(2)
        panel_layout.addWidget(self.cmb_sizemode)
        
        panel_layout.addWidget(QLabel("■ 실시간 계산 결과"))
        self.lbl_memory_calc = QLabel("이론 메모리: -")
        self.lbl_memory_calc.setStyleSheet("font-weight: bold; color: #2cb67d;")
        panel_layout.addWidget(self.lbl_memory_calc)
        
        self.lbl_target_fps = QLabel("목표 FPS: -")
        self.lbl_target_fps.setStyleSheet("font-weight: bold; color: #2cb67d;")
        panel_layout.addWidget(self.lbl_target_fps)
        
        self.lbl_measured_fps = QLabel("실제 측정 FPS: 0 (정지됨)")
        self.lbl_measured_fps.setStyleSheet("font-weight: bold; color: #ef4565; font-size: 14px;")
        panel_layout.addWidget(self.lbl_measured_fps)
        
        # Start/Stop Buttons
        btn_box = QHBoxLayout()
        self.btn_start = QPushButton("시작 (Start)")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start_timer)
        self.btn_stop = QPushButton("중지 (Stop)")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_timer)
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_stop)
        panel_layout.addLayout(btn_box)
        
        # Principle text
        txt_principle = QTextEdit()
        txt_principle.setReadOnly(True)
        txt_principle.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 이미지 크기 (Memory Size)\n"
            "   공식: 가로 × 세로 × 채널 수 (Bytes)\n"
            "   - 그레이스케일: 1채널 (명암값)\n"
            "   - 컬러(BGR): 3채널 (Blue, Green, Red)\n\n"
            "2. FPS 와 Interval 관계\n"
            "   공식: Target FPS = 1000 / Interval(ms)\n"
            "   - 33ms ≒ 30.3 FPS (표준 비디오)\n"
            "   - 16ms ≒ 62.5 FPS\n\n"
            "3. 렌더링 오차\n"
            "   - PC 성능, 화면 크기 조절, UI 타이머 오차로 인해 실제 측정 FPS는 계산값보다 작을 수 있습니다."
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
            self.lbl_q1_result.setText("정답! 1280 × 720 × 3 = 2,764,800 바이트(약 2.64MB)입니다. 컴퓨터가 기억해야 할 점의 용량이죠!")
        elif self.rdo_q1_x.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q1_result.setText("오답입니다. 가로(1280) × 세로(720) × 컬러겹수(3) 공식을 다시 계산해 보세요!")
        else:
            self.lbl_q1_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q1_result.setText("답안을 먼저 체크해 주세요.")
            
        # Q2 Check (Answer: O)
        if self.rdo_q2_o.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q2_result.setText("정답! StretchImage는 비율을 무시하고 액자에 맞추므로 화면이 찌그러져 보일 수 있어요.")
        elif self.rdo_q2_x.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q2_result.setText("오답입니다. 비율을 예쁘게 유지하면서 화면 여백을 두는 모드는 'Zoom' 모드입니다.")
        else:
            self.lbl_q2_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q2_result.setText("답안을 먼저 체크해 주세요.")

    def get_resolution(self):
        idx = self.cmb_resolution.currentIndex()
        if idx == 0: return 320, 240
        elif idx == 2: return 1280, 720
        elif idx == 3: return 1920, 1080
        else: return 640, 480
        
    def get_channels(self):
        return 1 if self.cmb_channels.currentIndex() == 0 else 3

    def update_calculations(self):
        w, h = self.get_resolution()
        ch = self.get_channels()
        num_bytes = w * h * ch
        mb = num_bytes / (1024.0 * 1024.0)
        self.lbl_memory_calc.setText(f"이론 메모리: {num_bytes:,} B ({mb:.2f} MB)")
        
        interval = self.nud_interval.value()
        self.lbl_target_fps.setText(f"목표 FPS: {1000.0 / interval:.1f} (주기 {interval}ms)")

    def interval_changed(self):
        self.update_calculations()
        if self.render_timer.isActive():
            self.render_timer.setInterval(self.nud_interval.value())

    def start_timer(self):
        self.frame_counter = 0
        self.fps_timer.start()
        self.render_timer.start(self.nud_interval.value())
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cmb_resolution.setEnabled(False)
        self.cmb_channels.setEnabled(False)
        
    def stop_timer(self):
        self.render_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cmb_resolution.setEnabled(True)
        self.cmb_channels.setEnabled(True)
        self.lbl_measured_fps.setText("실제 측정 FPS: 0 (정지됨)")

    def render_timer_tick(self):
        w, h = self.get_resolution()
        ch = self.get_channels()
        
        # 1. Create base NumPy array
        if ch == 1:
            img = np.zeros((h, w), dtype=np.uint8)
            img[:] = 40 # Dark Gray background
            ball_color = 200 # Light Gray ball
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # Fill with (B=40, G=30, R=20)
            img[:] = (40, 30, 20)
            ball_color = (0, 69, 255) # OrangeRed ball (BGR order: B=0, G=69, R=255)
            
        # 2. Update ball physics based on resolution scale
        scale_x = w / 640.0
        scale_y = h / 480.0
        
        self.ball_x += self.ball_speed_x * scale_x
        self.ball_y += self.ball_speed_y * scale_y
        r = max(10.0, self.ball_radius * scale_x)
        
        # Boundary collision
        if self.ball_x - r < 0:
            self.ball_x = r
            self.ball_speed_x = -self.ball_speed_x
        elif self.ball_x + r > w:
            self.ball_x = w - r
            self.ball_speed_x = -self.ball_speed_x
            
        if self.ball_y - r < 0:
            self.ball_y = r
            self.ball_speed_y = -self.ball_speed_y
        elif self.ball_y + r > h:
            self.ball_y = h - r
            self.ball_speed_y = -self.ball_speed_y
            
        # 3. Draw circle
        center = (int(self.ball_x), int(self.ball_y))
        cv2.circle(img, center, int(r), ball_color, -1)
        
        # 4. Draw crosshairs
        grid_color = 120 if ch == 1 else (150, 150, 150)
        cv2.line(img, (w // 2, 0), (w // 2, h), grid_color, 1)
        cv2.line(img, (0, h // 2), (w, h // 2), grid_color, 1)
        
        # 5. Draw HUD texts
        font_scale = max(0.4, w / 800.0)
        text_color = 255 if ch == 1 else (255, 255, 255)
        thickness = max(1, int(w / 640.0))
        
        cv2.putText(img, f"Simulated Frame: {w} x {h} ({ch}Ch)", (20, int(30 * scale_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.putText(img, time.strftime("%H:%M:%S"), (20, int(60 * scale_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.putText(img, f"Memory: {w * h * ch / 1024.0 / 1024.0:.2f} MB", (20, int(90 * scale_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        # 6. Apply SizeMode and show
        disp = apply_size_mode(img, 720, 480, self.cmb_sizemode.currentText())
        pix = ndarray_to_qpixmap(disp)
        self.pic_box.setPixmap(pix)
        
        # 7. FPS calculation
        self.frame_counter += 1
        elapsed = self.fps_timer.elapsed()
        if elapsed >= 1000:
            measured_fps = self.frame_counter / (elapsed / 1000.0)
            self.lbl_measured_fps.setText(f"실제 측정 FPS: {measured_fps:.1f}")
            self.frame_counter = 0
            self.fps_timer.restart()
            
    def closeEvent(self, event):
        self.stop_timer()
        super().closeEvent(event)
