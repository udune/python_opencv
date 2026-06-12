import os
import sys
import time
from PyQt6.QtCore import QTimer, Qt, QTime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QGroupBox, QRadioButton, QPushButton, QComboBox, 
    QSpinBox, QSlider, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap
import cv2
import numpy as np

from ui_helpers import apply_size_mode, ndarray_to_qpixmap

class FormCh04(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH04 - 동영상 파일 출력 (재생/저장) 학습 및 실습")
        self.resize(1200, 740)
        self.setMinimumSize(1000, 700)
        
        # Playback state variables
        self.play_capture = None
        self.temp_video_path = None
        self.play_total_frames = 0
        self.play_fps = 30.0
        self.is_tracking = False
        
        # Recording state variables
        self.rec_capture = None
        self.rec_writer = None
        self.rec_frame_count = 0
        self.rec_file_path = ""
        self.rec_w = 640
        self.rec_h = 480
        
        # Timers
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_timer_tick)
        
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self.rec_timer_tick)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabs = QTabWidget()
        self.tab_theory = QWidget()
        self.tab_playback = QWidget()
        self.tab_record = QWidget()
        
        self.tabs.addTab(self.tab_theory, "📚 1. 핵심 이론 및 자가진단 퀴즈")
        self.tabs.addTab(self.tab_playback, "🧪 2. 동영상 파일 재생 (Playback Lab)")
        self.tabs.addTab(self.tab_record, "🧪 3. 동영상 파일 저장 (Record Lab)")
        
        main_layout.addWidget(self.tabs)
        
        self.init_theory_tab()
        self.init_playback_tab()
        self.init_record_tab()
        
    def init_theory_tab(self):
        layout = QHBoxLayout(self.tab_theory)
        
        # Left side: Theory & Quiz
        left_layout = QVBoxLayout()
        
        lbl_title = QLabel("004. 동영상 재생(디코딩) & 저장(인코딩) 핵심 이론 학습")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f5af0; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)
        
        txt_theory = QTextEdit()
        txt_theory.setReadOnly(True)
        txt_theory.setPlainText(
            "■ [동영상 압축 풀기 및 재생 (cv2.VideoCapture)]\n"
            "우리가 흔히 보는 동영상 파일(mp4, avi 등)은 용량을 줄이기 위해 수천 장의 사진을 꽉꽉 압축해서 '비디오 파일 상자'에 차곡차곡 포장해 둔 것입니다.\n"
            "  - 비디오 상자 열기: cap = cv2.VideoCapture(파일경로)\n"
            "  - 이 도구는 동영상 파일을 열어 찌그러진 사진들을 원래대로 깨끗하게 펼쳐서 보여줄 준비를 해 줘요.\n"
            "  - 재생 원리: 타이머 주기에 맞춰 read()를 계속 누르면, 압축이 풀린 깨끗한 사진 한 장(ndarray)이 순서대로 쏙 튀어나오고, 이를 QPixmap으로 변환해 우리 눈앞에 연속으로 띄워 주는 것이랍니다.\n\n"
            "※ 원하는 장면으로 순식간에 점프하기 (Seek):\n"
            "  - 슬라이더를 잡고 끌어당기면, `cap.set(cv2.CAP_PROP_POS_FRAMES, 프레임번호)`을 통해 동영상 헤드를 원하는 사진 번호의 위치로 한순간에 이동(Seek)시킬 수 있어요.\n\n"
            "--------------------------------------------------\n\n"
            "■ [사진들을 엮어서 비디오 파일로 포장하기 (cv2.VideoWriter)]\n"
            "카메라로 찍는 연속 사진들을 조립해서 하나의 멋진 동영상 파일로 저장하는 기술이에요.\n"
            "  - 저장 준비: writer = cv2.VideoWriter(저장경로, 코덱종류, FPS, 해상도크기)\n"
            "    - 코덱(fourcc): XVID, MJPG 등 어떤 방식으로 영상을 압축해 상자에 넣을지 고르는 암호 기법이에요.\n"
            "    - FPS와 크기: 완성될 동영상의 부드러움(예: 30 FPS)과 화면 크기를 미리 약속해요.\n"
            "  - 사진 밀어넣기: writer.write(frame) 함수를 통해 매 순간 찰칵 찍은 사진들을 인코더 상자에 하나씩 주입해 차곡차곡 쌓아 올립니다.\n\n"
            "★ 가장 중요한 주의사항: 다 만들었으면 꼭 뚜껑 닫기 (release)\n"
            "  - 녹화가 다 끝났다면 반드시 `writer.release()`를 불러서 비디오 상자 뚜껑을 꽉 닫아 봉인해야 해요.\n"
            "  - 이때 비디오 파일 내부의 머리말(재생 시간, 규격 등) 정보가 최종적으로 기록되는데, 만약 닫지 않고 강제 종료하면 컴퓨터가 저장을 마무리 짓지 못해 결국 재생할 수 없는 빈 깡통 파일(0바이트)이 남게 됩니다."
        )
        txt_theory.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 10px;")
        left_layout.addWidget(txt_theory)
        
        # Quiz Area
        grp_quiz = QGroupBox("✍ 자가 진단 퀴즈 (이론 검증)")
        quiz_layout = QVBoxLayout(grp_quiz)
        
        # Quiz 1
        q1_group = QGroupBox("질문 1. 동영상 원하는 장면으로 점프하기 (Seek)")
        q1_layout = QVBoxLayout(q1_group)
        q1_lbl = QLabel("동영상을 보다가 슬라이더(트랙바)를 조작해 내가 원하는 장면이나 시간대로 즉시 이동시키는 기능(Seek)은 OpenCV에서 특정 프레임 위치 속성을 세팅해 손쉽게 구현할 수 있다.")
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
        q2_group = QGroupBox("질문 2. VideoWriter 파일 저장 봉인")
        q2_layout = QVBoxLayout(q2_group)
        q2_lbl = QLabel("비디오 파일에 사진을 엮어서 저장할 때는, 다 쓰고 나서 release()로 뚜껑을 닫지 않고 프로그램을 그냥 강제 종료해도 비디오 파일이 정상적으로 열리고 잘 재생된다.")
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
        
        btn_goto_play = QPushButton("실습 실험실로 이동하기 (Go to Playback Lab) ▶")
        btn_goto_play.setObjectName("btnGoToLab")
        btn_goto_play.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        quiz_layout.addWidget(btn_goto_play)
        
        left_layout.addWidget(grp_quiz)
        layout.addLayout(left_layout, 1)
        
        # Right side: Diagram
        right_layout = QVBoxLayout()
        lbl_diag_title = QLabel("🖼 시각 자료: 동영상 생명 주기 (인코딩 및 디코딩)")
        lbl_diag_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f5af0;")
        right_layout.addWidget(lbl_diag_title)
        
        self.pic_diagram = QLabel()
        self.pic_diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_diagram.setStyleSheet("background-color: #101012; border: 1px solid #2d2d30; border-radius: 6px;")
        
        # Load ch04_video_lifecycle_ko.png
        img_path = self.get_resource_path("ch04_video_lifecycle_ko.png")
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.pic_diagram.setPixmap(pix.scaled(580, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pic_diagram.setText("[이미지 파일 없음: ch04_video_lifecycle_ko.png]")
            
        right_layout.addWidget(self.pic_diagram)
        
        txt_diag_desc = QTextEdit()
        txt_diag_desc.setReadOnly(True)
        txt_diag_desc.setPlainText(
            "【인포그래픽 설명】\n"
            "1. 동영상 디코딩(재생) 파이프라인: 파일로부터 데이터를 로드(cv2.VideoCapture)하여 디코더를 초기화합니다. 이후 프레임을 순차적으로 읽고(read), QPixmap으로 변환해 화면에 렌더링(Show)하는 주기를 거칩니다. CAP_PROP_POS_FRAMES 제어로 임의의 프레임 위치로 이동(Seek)할 수도 있습니다.\n"
            "2. 동영상 인코딩(저장) 파이프라인: 저장 대상 파일명, 코덱(FourCC), FPS, 프레임 해상도를 지정해 VideoWriter 인코더를 초기화합니다. 가공 및 리사이징된 이미지를 스트림에 연속으로 기록(write)하고, 녹화가 끝나면 반드시 해제(release)를 거쳐 완성합니다.\n"
            "3. 주의사항: 인코더를 수동으로 닫아주지(release) 않으면, 비디오 파일의 메타데이터와 프레임 인덱스 목록이 올바르게 하드디스크에 써지지 않아 재생이 불가능한 손상된 0바이트 파일이 남습니다."
        )
        txt_diag_desc.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 5px;")
        txt_diag_desc.setMaximumHeight(150)
        right_layout.addWidget(txt_diag_desc)
        
        layout.addLayout(right_layout, 1)

    def init_playback_tab(self):
        layout = QHBoxLayout(self.tab_playback)
        
        # Left Side: Player Screen + Seek Bar + Info
        left_layout = QVBoxLayout()
        self.pb_play = QLabel()
        self.pb_play.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pb_play.setStyleSheet("background-color: #000000; border: 1px solid #2d2d30; border-radius: 6px;")
        self.pb_play.setFixedSize(700, 420)
        left_layout.addWidget(self.pb_play)
        
        self.track_frame = QSlider(Qt.Orientation.Horizontal)
        self.track_frame.setEnabled(False)
        self.track_frame.sliderPressed.connect(self.slider_pressed)
        self.track_frame.sliderReleased.connect(self.slider_released)
        self.track_frame.sliderMoved.connect(self.slider_moved)
        left_layout.addWidget(self.track_frame)
        
        self.lbl_play_frame_info = QLabel("프레임: 0 / 0 (00:00:00 / 00:00:00)")
        self.lbl_play_frame_info.setStyleSheet("font-weight: bold; font-size: 14px; color: #2cb67d;")
        left_layout.addWidget(self.lbl_play_frame_info)
        layout.addLayout(left_layout)
        
        # Right Side: Playback Controls Panel
        panel = QGroupBox("재생 실습 제어")
        panel_layout = QVBoxLayout(panel)
        
        # 1. Open File
        panel_layout.addWidget(QLabel("1. 동영상 파일 선택"))
        btn_open = QPushButton("파일 열기 (Open File)")
        btn_open.setObjectName("btnPlayOpenFile")
        btn_open.clicked.connect(self.btn_play_open_file_clicked)
        panel_layout.addWidget(btn_open)
        
        # 2. Control Buttons
        panel_layout.addWidget(QLabel("2. 재생 제어"))
        btn_box = QHBoxLayout()
        self.btn_play = QPushButton("재생 (Play)")
        self.btn_play.setObjectName("btnPlayStart")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.start_playback)
        
        self.btn_pause = QPushButton("일시정지")
        self.btn_pause.setObjectName("btnPlayPause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_playback)
        
        self.btn_stop = QPushButton("정지 (Stop)")
        self.btn_stop.setObjectName("btnPlayStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        
        btn_box.addWidget(self.btn_play)
        btn_box.addWidget(self.btn_pause)
        btn_box.addWidget(self.btn_stop)
        panel_layout.addLayout(btn_box)
        
        # 3. Playback speed
        panel_layout.addWidget(QLabel("3. 재생 속도 조절 (Interval, ms)"))
        self.nud_play_interval = QSpinBox()
        self.nud_play_interval.setRange(5, 500)
        self.nud_play_interval.setValue(33)
        self.nud_play_interval.valueChanged.connect(self.play_interval_changed)
        panel_layout.addWidget(self.nud_play_interval)
        
        # Status box
        panel_layout.addWidget(QLabel("■ 재생 정보 및 상태"))
        self.lbl_play_status = QLabel("상태: 파일 대기 중")
        self.lbl_play_status.setWordWrap(True)
        self.lbl_play_status.setStyleSheet("background-color: #2c2c1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        self.lbl_play_status.setMinimumHeight(55)
        panel_layout.addWidget(self.lbl_play_status)
        
        # Principle text
        txt_principle = QTextEdit()
        txt_principle.setReadOnly(True)
        txt_principle.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 동영상 읽기: CvCapture.FromFile(path)\n"
            "   - 동영상 컨테이너(mp4, avi)를 열어 디코딩합니다.\n"
            "2. 재생 원리: Timer 루프 + QueryFrame()\n"
            "   - 지정된 Interval마다 디코더로부터 프레임을 가져와 비트맵으로 렌더링합니다.\n"
            "3. 위치 이동(Seek): SetCaptureProperty(..)\n"
            "   - CaptureProperty.PosFrames 속성을 수정해 재생 헤드를 임의의 프레임 위치로 이동시킵니다."
        )
        txt_principle.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 4px; padding: 5px;")
        panel_layout.addWidget(txt_principle)
        
        layout.addWidget(panel)

    def init_record_tab(self):
        layout = QHBoxLayout(self.tab_record)
        
        # Left Side: Camera Preview Screen
        left_layout = QVBoxLayout()
        self.pb_rec = QLabel()
        self.pb_rec.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pb_rec.setStyleSheet("background-color: #000000; border: 1px solid #2d2d30; border-radius: 6px;")
        self.pb_rec.setFixedSize(700, 420)
        left_layout.addWidget(self.pb_rec)
        
        txt_guide = QTextEdit()
        txt_guide.setReadOnly(True)
        txt_guide.setPlainText("실습 가이드: 녹화 시작 시 대화상자에서 저장할 경로를 정한 후 녹화를 시작할 수 있습니다.")
        txt_guide.setStyleSheet("background-color: #2b2b1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        txt_guide.setMaximumHeight(50)
        left_layout.addWidget(txt_guide)
        layout.addLayout(left_layout)
        
        # Right Side: Recording Panel
        panel = QGroupBox("녹화 실습 설정")
        panel_layout = QVBoxLayout(panel)
        
        # 1. Camera Selector
        panel_layout.addWidget(QLabel("1. 카메라 선택 (Index)"))
        self.nud_rec_camera_index = QSpinBox()
        self.nud_rec_camera_index.setRange(0, 5)
        self.nud_rec_camera_index.setValue(0)
        panel_layout.addWidget(self.nud_rec_camera_index)
        
        # 2. Recording Resolution
        panel_layout.addWidget(QLabel("2. 녹화 해상도 (Resolution)"))
        self.cmb_rec_resolution = QComboBox()
        self.cmb_rec_resolution.addItems(["320 x 240", "640 x 480"])
        self.cmb_rec_resolution.setCurrentIndex(1)
        panel_layout.addWidget(self.cmb_rec_resolution)
        
        # 3. Codec
        panel_layout.addWidget(QLabel("3. 비디오 코덱 (Codec)"))
        self.cmb_rec_codec = QComboBox()
        self.cmb_rec_codec.addItems(["XVID (AVI)", "DIB / Raw (AVI)", "MJPG (AVI)"])
        self.cmb_rec_codec.setCurrentIndex(2)
        panel_layout.addWidget(self.cmb_rec_codec)
        
        # 4. Start/Stop Record
        panel_layout.addWidget(QLabel("4. 녹화 시작 / 종료"))
        btn_box = QHBoxLayout()
        self.btn_rec_start = QPushButton("녹화 시작 (Record)")
        self.btn_rec_start.setObjectName("btnRecStart")
        self.btn_rec_start.clicked.connect(self.start_recording)
        
        self.btn_rec_stop = QPushButton("녹화 중지 (Stop)")
        self.btn_rec_stop.setObjectName("btnRecStop")
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_stop.clicked.connect(self.btn_rec_stop_clicked)
        
        btn_box.addWidget(self.btn_rec_start)
        btn_box.addWidget(self.btn_rec_stop)
        panel_layout.addLayout(btn_box)
        
        # Status
        panel_layout.addWidget(QLabel("■ 녹화 진행 상태"))
        self.lbl_rec_status = QLabel("상태: 녹화 준비 완료")
        self.lbl_rec_status.setWordWrap(True)
        self.lbl_rec_status.setStyleSheet("background-color: #2c2c1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        self.lbl_rec_status.setMinimumHeight(45)
        panel_layout.addWidget(self.lbl_rec_status)
        
        self.lbl_rec_info = QLabel("저장 프레임 수: 0 | 크기: 0 KB")
        self.lbl_rec_info.setStyleSheet("font-weight: bold; color: #2cb67d;")
        panel_layout.addWidget(self.lbl_rec_info)
        
        # Principle text
        txt_principle = QTextEdit()
        txt_principle.setReadOnly(True)
        txt_principle.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 파일 작성: CvVideoWriter(path, codec, fps, size) [cv2.VideoWriter]\n"
            "   - 지정된 코덱(예: MJPG, XVID), 속도, 크기로 출력 동영상 스트림 파일을 초기화합니다.\n"
            "2. 프레임 주입: WriteFrame(iplImage) [writer.write(frame)]\n"
            "   - 매 캡처 프레임을 라이터 객체에 써넣습니다.\n"
            "3. 최종 저장: Dispose() 필수! [writer.release()]\n"
            "   - 라이터 객체를 닫을 때 파일 헤더 정보와 인덱스가 완전하게 기록되어 재생 가능한 정상 비디오가 됩니다."
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
            self.lbl_q1_result.setText("정답! `cv2.CAP_PROP_POS_FRAMES` 속성 설정을 통해 원하는 특정 프레임 사진 번호 위치로 재생 헤드를 즉시 이동시킬 수 있어요.")
        elif self.rdo_q1_x.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q1_result.setText("오답입니다. 슬라이더 위치에 맞추어 재생 헤드를 강제 이동(Seek)시키는 구조는 동영상 제어의 기본이랍니다.")
        else:
            self.lbl_q1_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q1_result.setText("답안을 먼저 체크해 주세요.")
            
        # Q2 Check (Answer: X)
        if self.rdo_q2_x.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q2_result.setText("정답! 녹화 후 `release()`를 깜빡하면 동영상 상자의 뚜껑이 안 닫혀서 최종 기록 정보가 누락되고 망가진 0바이트 파일이 남아요.")
        elif self.rdo_q2_o.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q2_result.setText("오답입니다. 스트림 마무리 처리인 release()가 누락되면 동영상 규격이 깨져 재생할 수 없게 된답니다.")
        else:
            self.lbl_q2_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q2_result.setText("답안을 먼저 체크해 주세요.")

    # =====================================================================
    # Playback Methods
    # =====================================================================
    def btn_play_open_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "재생할 동영상 파일 선택", "", 
            "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)
            
    def load_video(self, file_path):
        try:
            self.stop_playback()
            
            # Check for non-ASCII characters in the path (workaround for OpenCV Windows VideoCapture)
            pathToOpen = file_path
            if any(ord(c) > 127 for c in file_path):
                ext = os.path.splitext(file_path)[1]
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.temp_video_path = os.path.join(base_dir, f"temp_video_playback{ext}")
                
                if os.path.exists(self.temp_video_path):
                    try:
                        os.remove(self.temp_video_path)
                    except Exception:
                        pass
                        
                import shutil
                try:
                    shutil.copy2(file_path, self.temp_video_path)
                    pathToOpen = self.temp_video_path
                except Exception as e:
                    QMessageBox.warning(self, "임시 파일 생성 실패", f"한글 경로 우회를 위한 임시 파일 생성에 실패했습니다:\n{str(e)}")
                    return

            # Open Video Capture
            self.play_capture = cv2.VideoCapture(pathToOpen)
            if not self.play_capture.isOpened():
                QMessageBox.warning(self, "비디오 로드 에러", "비디오를 로드하는 데 실패했습니다.")
                self.play_capture = None
                self.cleanup_temp_file()
                return
                
            # Verify we can read a frame
            ret, frame = self.play_capture.read()
            if not ret or frame is None:
                self.play_capture.release()
                self.play_capture = None
                QMessageBox.warning(self, "비디오 디코딩 실패", "동영상 첫 프레임을 읽을 수 없습니다.")
                return
                
            # Reset to frame 0
            self.play_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Fetch metadata
            self.play_total_frames = int(self.play_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.play_fps = self.play_capture.get(cv2.CAP_PROP_FPS)
            if self.play_fps <= 0:
                self.play_fps = 30.0
                
            # Configure UI
            interval_ms = max(5, min(500, int(1000.0 / self.play_fps)))
            self.nud_play_interval.setValue(interval_ms)
            self.play_timer.setInterval(interval_ms)
            
            self.track_frame.setRange(0, max(0, self.play_total_frames - 1))
            self.track_frame.setValue(0)
            self.track_frame.setEnabled(True)
            
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(True)
            
            filename = os.path.basename(file_path)
            self.lbl_play_status.setText(f"파일명: {filename}\nFPS: {self.play_fps:.2f} / 총 {self.play_total_frames} 프레임")
            self.update_play_frame_info(0)
            self.show_frame(0)
            
        except Exception as ex:
            self.stop_playback()
            QMessageBox.critical(self, "오류", f"비디오 로딩 실패: {str(ex)}")

    def show_frame(self, frame_index):
        if self.play_capture is None:
            return
        self.play_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.play_capture.read()
        if ret and frame is not None:
            disp = apply_size_mode(frame, 700, 420, "Zoom") # Hardcode Zoom mode for beautiful layout
            pix = ndarray_to_qpixmap(disp)
            self.pb_play.setPixmap(pix)

    def update_play_frame_info(self, current_frame):
        cur_sec = current_frame / self.play_fps
        total_sec = self.play_total_frames / self.play_fps
        
        cur_time = QTime(0, 0, 0).addSecs(int(cur_sec))
        tot_time = QTime(0, 0, 0).addSecs(int(total_sec))
        
        cur_str = cur_time.toString("hh:mm:ss")
        tot_str = tot_time.toString("hh:mm:ss")
        
        self.lbl_play_frame_info.setText(f"프레임: {current_frame} / {self.play_total_frames} ({cur_str} / {tot_str})")

    def play_interval_changed(self):
        if self.play_timer.isActive():
            self.play_timer.setInterval(self.nud_play_interval.value())

    def start_playback(self):
        if self.play_capture is None:
            return
        self.play_timer.start(self.nud_play_interval.value())
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_play_status.setText("상태: 재생 중")

    def pause_playback(self):
        self.play_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.lbl_play_status.setText("상태: 일시 정지")

    def stop_playback(self):
        self.play_timer.stop()
        if self.play_capture is not None:
            self.play_capture.release()
            self.play_capture = None
            
        self.cleanup_temp_file()
        self.pb_play.clear()
        self.track_frame.setValue(0)
        self.track_frame.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_play_status.setText("상태: 정지됨")
        self.lbl_play_frame_info.setText("프레임: 0 / 0 (00:00:00 / 00:00:00)")

    def play_timer_tick(self):
        if self.play_capture is None or self.is_tracking:
            return
            
        # Get current frame index before read
        current_frame = int(self.play_capture.get(cv2.CAP_PROP_POS_FRAMES))
        
        ret, frame = self.play_capture.read()
        if not ret or frame is None:
            self.stop_playback()
            self.lbl_play_status.setText("상태: 재생 완료")
            return
            
        disp = apply_size_mode(frame, 700, 420, "Zoom")
        pix = ndarray_to_qpixmap(disp)
        self.pb_play.setPixmap(pix)
        
        self.track_frame.setValue(min(current_frame + 1, self.track_frame.maximum()))
        self.update_play_frame_info(current_frame + 1)
        
        if self.play_total_frames > 0 and (current_frame + 1) >= self.play_total_frames:
            self.stop_playback()
            self.lbl_play_status.setText("상태: 재생 완료")

    def slider_pressed(self):
        self.is_tracking = True

    def slider_released(self):
        self.is_tracking = False
        if self.play_capture is not None:
            val = self.track_frame.value()
            self.show_frame(val)
            self.update_play_frame_info(val)

    def slider_moved(self, val):
        if self.play_capture is not None:
            self.show_frame(val)
            self.update_play_frame_info(val)

    # =====================================================================
    # Recording Methods
    # =====================================================================
    def start_recording(self):
        try:
            self.stop_recording()
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "저장할 동영상 파일 명 지정", "output_record.avi", 
                "AVI Video File (*.avi)"
            )
            if not file_path:
                return
            self.rec_file_path = file_path
            
            # Start camera
            idx = self.nud_rec_camera_index.value()
            self.rec_capture = cv2.VideoCapture(idx)
            if not self.rec_capture.isOpened():
                QMessageBox.warning(self, "카메라 연결 실패", "카메라 장치를 열 수 없습니다.")
                self.rec_capture = None
                return
                
            # Set resolution
            self.rec_w = 320 if self.cmb_rec_resolution.currentIndex() == 0 else 640
            self.rec_h = 240 if self.cmb_rec_resolution.currentIndex() == 0 else 480
            self.rec_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.rec_w)
            self.rec_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rec_h)
            
            # Verify camera grab
            ret, frame = self.rec_capture.read()
            if not ret or frame is None:
                self.rec_capture.release()
                self.rec_capture = None
                QMessageBox.warning(self, "오류", "카메라 영상 획득에 실패했습니다.")
                return
                
            # Configure Codec
            codec_idx = self.cmb_rec_codec.currentIndex()
            if codec_idx == 0:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif codec_idx == 1:
                # Raw/Uncompressed codec - fourcc 'DIB ' is uncompressed RGB
                fourcc = cv2.VideoWriter_fourcc(*'DIB ')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                
            self.rec_writer = cv2.VideoWriter(self.rec_file_path, fourcc, 30.0, (self.rec_w, self.rec_h))
            if not self.rec_writer.isOpened():
                self.rec_capture.release()
                self.rec_capture = None
                self.rec_writer = None
                QMessageBox.warning(self, "오류", "동영상 인코더를 실행할 수 없습니다.")
                return
                
            self.rec_frame_count = 0
            self.rec_timer.start(33) # ~30 FPS
            
            self.btn_rec_start.setEnabled(False)
            self.btn_rec_stop.setEnabled(True)
            self.nud_rec_camera_index.setEnabled(False)
            self.cmb_rec_resolution.setEnabled(False)
            self.cmb_rec_codec.setEnabled(False)
            
            filename = os.path.basename(self.rec_file_path)
            self.lbl_rec_status.setText(f"녹화 진행 중: {filename}")
            
        except Exception as ex:
            self.stop_recording()
            QMessageBox.critical(self, "녹화 초기화 오류", f"에러: {str(ex)}")

    def btn_rec_stop_clicked(self):
        saved_path = self.rec_file_path
        self.stop_recording()
        QMessageBox.information(self, "녹화 완료", f"녹화가 종료되었습니다.\n파일 저장 위치:\n{saved_path}")

    def stop_recording(self):
        self.rec_timer.stop()
        
        if self.rec_writer is not None:
            self.rec_writer.release()
            self.rec_writer = None
            
        if self.rec_capture is not None:
            self.rec_capture.release()
            self.rec_capture = None
            
        self.pb_rec.clear()
        
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)
        self.nud_rec_camera_index.setEnabled(True)
        self.cmb_rec_resolution.setEnabled(True)
        self.cmb_rec_codec.setEnabled(True)
        self.lbl_rec_status.setText("상태: 녹화 종료됨")

    def rec_timer_tick(self):
        if self.rec_capture is None or self.rec_writer is None:
            return
            
        ret, frame = self.rec_capture.read()
        if not ret or frame is None:
            return
            
        # Ensure dimensions match
        h, w = frame.shape[:2]
        if w != self.rec_w or h != self.rec_h:
            frame_resized = cv2.resize(frame, (self.rec_w, self.rec_h))
        else:
            frame_resized = frame
            
        # Write to video file
        self.rec_writer.write(frame_resized)
        
        # Display preview
        disp = apply_size_mode(frame_resized, 700, 420, "Zoom")
        pix = ndarray_to_qpixmap(disp)
        self.pb_rec.setPixmap(pix)
        
        # Update progress labels
        self.rec_frame_count += 1
        file_size_kb = 0
        if os.path.exists(self.rec_file_path):
            file_size_kb = os.path.getsize(self.rec_file_path) / 1024
            
        self.lbl_rec_info.setText(f"저장 프레임 수: {self.rec_frame_count} | 크기: {file_size_kb:,.0f} KB")

    def cleanup_temp_file(self):
        if hasattr(self, 'temp_video_path') and self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
            except Exception:
                pass
            self.temp_video_path = None

    def closeEvent(self, event):
        self.stop_playback()
        self.stop_recording()
        self.cleanup_temp_file()
        super().closeEvent(event)
