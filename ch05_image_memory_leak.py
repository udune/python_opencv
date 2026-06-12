import os
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QGroupBox, QRadioButton, QPushButton, QComboBox, 
    QProgressBar, QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtGui import QFont, QPixmap
import cv2
import numpy as np

from ui_helpers import apply_size_mode, ndarray_to_qpixmap

class FormCh05(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH05 - 이미지 출력 & 메모리 누수 학습 및 실습")
        self.resize(1200, 740)
        self.setMinimumSize(1000, 700)
        
        # State variables
        self.selected_file_path = ""
        self.active_image = None
        self.leaked_images = []
        self.active_allocated_bytes = 0
        self.total_leaked_bytes = 0
        self.max_bytes = 10 * 1024 * 1024  # 10 MB limit for the visualizer progress bar
        
        self.init_ui()
        self.update_memory_gauges()
        
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
        
        lbl_title = QLabel("005. 이미지 출력 & 네이티브 메모리 관리 핵심 이론 학습")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f5af0; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)
        
        txt_theory = QTextEdit()
        txt_theory.setReadOnly(True)
        txt_theory.setPlainText(
            "■ [파이썬의 청소 로봇과 이미지 메모리]\n"
            "파이썬(Python)은 메모리를 정리해 주는 영리한 **'청소 로봇(가비지 컬렉터 - GC)'**을 내장하고 있어 직접 일일이 청소할 필요가 없어 무척 편리해요.\n"
            "OpenCV로 읽어온 이미지 데이터(ndarray)는 헬륨 풍선과 같고, 이미지 변수(`src`)는 풍선 끈을 쥐고 있는 손가락과 같습니다.\n\n"
            "  - 풍선 끈 개수 추적 (참조 횟수 계산):\n"
            "    파이썬 청소 로봇은 각각의 이미지 풍선에 끈이 몇 개 묶여 있는지(Reference Count) 늘 확인합니다.\n"
            "    만약 끈을 쥐고 있던 손가락들이 모두 끈을 놓아버려 끈 개수가 0이 되면(참조 횟수 0), 청소 로봇이 '아, 이건 이제 안 쓰는 쓰레기구나!' 하고 풍선을 즉시 수거해 터트려 버립니다(메모리 자동 회수).\n"
            "  - 수동 해제 불필요: 따라서 파이썬에서는 C#이나 C++ 같은 다른 프로그래밍 언어와 달리 복잡하게 메모리 해제 명령을 직접 내릴 필요 없이, 변수에 비어있다는 뜻인 `None`을 대입해서 풍선 끈만 놓아 주면 즉시 메모리가 청소됩니다.\n\n"
            "--------------------------------------------------\n\n"
            "■ [방안이 풍선으로 꽉 차는 마법: 메모리 누수(Memory Leak)]\n"
            "  - `src = cv2.imread(...)`로 새 사진 풍선을 불어 손가락으로 쥐면 메모리를 차지합니다.\n"
            "  - 이 풍선 끈을 리스트(예: `leaked_list`)에 묶어놓은 상태에서, 내 손가락 변수(`src`)에 새로운 사진 풍선 끈을 쥐어 주면 어떻게 될까요?\n"
            "    1. 내 손은 새 풍선을 쥐었지만, 이전 풍선은 리스트에 여전히 끈이 묶여 있기 때문에 하늘로 날아가지(참조 횟수 0이 되지) 못합니다.\n"
            "    2. 결국 쓰지 않는 낡은 풍선들이 리스트에 묶여 방 한구석에 빽빽하게 쌓이게 됩니다.\n"
            "    3. 이 짓을 무한히 반복하면 방(컴퓨터 메모리)이 풍선으로 꽉 차서 더 이상 발 디딜 틈이 없어지고, 결국 프로그램이 숨이 막혀 뻗어버리는 현상을 **메모리 누수(Memory Leak)**라고 부릅니다.\n\n"
            "--------------------------------------------------\n\n"
            "■ [화면 액자 출력 후의 원본 관리]\n"
            "  - PyQt6 화면(QLabel)에 이미지를 출력하는 것은, 원본 풍선의 복사본 액자를 화면에 붙여두는 것과 같습니다.\n"
            "  - 화면에 액자가 붙었다고 해서 구석에 있는 진짜 무거운 원본 풍선 데이터가 소멸하는 것은 아니므로, 볼 일을 다 본 원본 변수는 꼭 `src = None`으로 끈을 끊어 줘야 방이 늘 깨끗하게 유지됩니다."
        )
        txt_theory.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 10px;")
        left_layout.addWidget(txt_theory)
        
        # Quiz Area
        grp_quiz = QGroupBox("✍ 자가 진단 퀴즈 (이론 검증)")
        quiz_layout = QVBoxLayout(grp_quiz)
        
        # Quiz 1
        q1_group = QGroupBox("질문 1. 파이썬의 자동 메모리 청소")
        q1_layout = QVBoxLayout(q1_group)
        q1_lbl = QLabel("파이썬 환경에서 어떤 이미지를 담고 있는 변수에 새로운 이미지를 대입해 덮어쓰면, 수동으로 삭제 함수를 부르지 않아도 다른 곳에 묶여있지 않다면 이전 이미지 메모리는 자동으로 즉시 회수된다.")
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
        q2_group = QGroupBox("질문 2. 화면 출력과 원본 자원 끈 관리")
        q2_layout = QVBoxLayout(q2_group)
        q2_lbl = QLabel("화면에 이미지를 성공적으로 출력했더라도, 원본 이미지 변수(src)의 참조 끈을 끊어주지(src = None) 않으면 메모리에서 소멸하지 않고 계속 살아있는다.")
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
        lbl_diag_title = QLabel("🖼 시각 자료: 네이티브 메모리 관리 & 메모리 누수 메커니즘")
        lbl_diag_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f5af0;")
        right_layout.addWidget(lbl_diag_title)
        
        self.pic_diagram = QLabel()
        self.pic_diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_diagram.setStyleSheet("background-color: #101012; border: 1px solid #2d2d30; border-radius: 6px;")
        
        # Load ch05_memory_leak_ko.png
        img_path = self.get_resource_path("ch05_memory_leak_ko.png")
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.pic_diagram.setPixmap(pix.scaled(580, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pic_diagram.setText("[이미지 파일 없음: ch05_memory_leak_ko.png]")
            
        right_layout.addWidget(self.pic_diagram)
        
        txt_diag_desc = QTextEdit()
        txt_diag_desc.setReadOnly(True)
        txt_diag_desc.setPlainText(
            "【인포그래픽 설명】\n"
            "1. 왼쪽(정상 메모리 해제 흐름): Python에서 cv2.imread로 ndarray를 생성하면 힙 영역에 데이터가 적재됩니다. 사용 완료 후 `src = None`으로 참조를 명시해 해제하면 참조 횟수가 0이 되어 파이썬 가비지 컬렉터에 의해 즉각 온전히 해제됩니다.\n"
            "2. 오른쪽(메모리 누수 발생 흐름): ndarray 데이터가 살아있는 상태에서 수동 참조 해제(None 대입) 없이 리스트 등에 보관하거나 덮어쓰기를 반복하면, 이전 배열의 참조가 누적되어 파이썬 GC가 메모리를 회수할 수 없게 됩니다.\n"
            "3. 핵심 요약: 컴퓨터 비전 처리 파이프라인에서 수백 FPS 속도로 대량의 이미지 할당/대입이 반복되므로, 참조 유실 관리를 누락하면 초 단위 내에 기가바이트 급의 메모리가 고갈되어 OOM(Out of Memory) 크래시를 맞닥뜨리게 됩니다."
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
        self.pic_box.setFixedSize(640, 480)
        left_layout.addWidget(self.pic_box)
        
        txt_guide = QTextEdit()
        txt_guide.setReadOnly(True)
        txt_guide.setPlainText(
            "실습 가이드: 파일 선택 후 [1단계: 로드] -> [2단계: 출력] 순서대로 실행하세요. "
            "3단계를 생략하고 1단계를 다시 실행하면 '메모리 누수'가 유발되는 현상을 관찰할 수 있습니다."
        )
        txt_guide.setStyleSheet("background-color: #2b2b1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        txt_guide.setMaximumHeight(50)
        left_layout.addWidget(txt_guide)
        layout.addLayout(left_layout)
        
        # Right side: Controls Panel
        panel = QGroupBox("005. 이미지 출력 & 메모리 누수 실험실")
        panel_layout = QVBoxLayout(panel)
        
        # 1. Image selection
        panel_layout.addWidget(QLabel("1. 이미지 파일 준비"))
        btn_box1 = QHBoxLayout()
        btn_select = QPushButton("이미지 선택 (Open File)")
        btn_select.clicked.connect(self.select_file)
        btn_default = QPushButton("기본 이미지 (Italia.jpg)")
        btn_default.clicked.connect(self.load_default_image)
        btn_box1.addWidget(btn_select)
        btn_box1.addWidget(btn_default)
        panel_layout.addLayout(btn_box1)
        
        self.lbl_file_path = QLabel("선택된 파일: 없음 (대기 중)")
        self.lbl_file_path.setStyleSheet("color: #72727a; font-size: 11px;")
        self.lbl_file_path.setWordWrap(True)
        panel_layout.addWidget(self.lbl_file_path)
        
        # 2. Step-by-Step interactive
        panel_layout.addWidget(QLabel("2. 단계별 실행 학습 (Interactive Steps)"))
        
        # Step 1
        step1_box = QHBoxLayout()
        self.btn_step1 = QPushButton("1단계: 로드")
        self.btn_step1.setObjectName("btnStep1")
        self.btn_step1.setEnabled(False)
        self.btn_step1.clicked.connect(self.btn_step1_clicked)
        step1_box.addWidget(self.btn_step1, 1)
        
        txt_code1 = QLabel("src = cv2.imread(path)")
        txt_code1.setStyleSheet("background-color: #1e1e24; color: #48c78e; font-family: 'Consolas'; border-radius: 4px; padding: 5px; font-size: 11px;")
        step1_box.addWidget(txt_code1, 2)
        panel_layout.addLayout(step1_box)
        
        # Step 2
        step2_box = QHBoxLayout()
        self.btn_step2 = QPushButton("2단계: 출력")
        self.btn_step2.setObjectName("btnStep2")
        self.btn_step2.setEnabled(False)
        self.btn_step2.clicked.connect(self.btn_step2_clicked)
        step2_box.addWidget(self.btn_step2, 1)
        
        txt_code2 = QLabel("show_image(src)")
        txt_code2.setStyleSheet("background-color: #1e1e24; color: #48c78e; font-family: 'Consolas'; border-radius: 4px; padding: 5px; font-size: 11px;")
        step2_box.addWidget(txt_code2, 2)
        panel_layout.addLayout(step2_box)
        
        # Step 3
        step3_box = QHBoxLayout()
        self.btn_step3 = QPushButton("3단계: 해제")
        self.btn_step3.setObjectName("btnStep3")
        self.btn_step3.setEnabled(False)
        self.btn_step3.clicked.connect(self.btn_step3_clicked)
        step3_box.addWidget(self.btn_step3, 1)
        
        txt_code3 = QLabel("src = None")
        txt_code3.setStyleSheet("background-color: #1e1e24; color: #48c78e; font-family: 'Consolas'; border-radius: 4px; padding: 5px; font-size: 11px;")
        step3_box.addWidget(txt_code3, 2)
        panel_layout.addLayout(step3_box)
        
        # 3. Memory Monitoring
        panel_layout.addWidget(QLabel("3. 실시간 네이티브 힙 모니터"))
        
        self.lbl_active_mem = QLabel("활성 네이티브 메모리: 0 Bytes (0.00 MB)")
        self.lbl_active_mem.setStyleSheet("font-size: 11px; font-weight: bold;")
        panel_layout.addWidget(self.lbl_active_mem)
        
        self.pb_active_mem = QProgressBar()
        self.pb_active_mem.setStyleSheet(
            "QProgressBar { background-color: #1e1e24; border: 1px solid #2d2d30; border-radius: 4px; height: 12px; text-align: center; color: transparent; }"
            "QProgressBar::chunk { background-color: #48c78e; border-radius: 3px; }"
        )
        self.pb_active_mem.setMaximum(self.max_bytes)
        panel_layout.addWidget(self.pb_active_mem)
        
        self.lbl_leaked_mem = QLabel("누출(누수) 메모리 합계: 0 Bytes (0.00 MB)")
        self.lbl_leaked_mem.setStyleSheet("font-size: 11px; font-weight: bold; color: #ef4565;")
        panel_layout.addWidget(self.lbl_leaked_mem)
        
        self.pb_leaked_mem = QProgressBar()
        self.pb_leaked_mem.setStyleSheet(
            "QProgressBar { background-color: #1e1e24; border: 1px solid #2d2d30; border-radius: 4px; height: 12px; text-align: center; color: transparent; }"
            "QProgressBar::chunk { background-color: #ef4565; border-radius: 3px; }"
        )
        self.pb_leaked_mem.setMaximum(self.max_bytes)
        panel_layout.addWidget(self.pb_leaked_mem)
        
        # Warnings and Reset buttons
        self.lbl_warning = QLabel("상태: 파일을 선택한 후 [1단계: 로드]를 실행하세요.")
        self.lbl_warning.setStyleSheet("background-color: #1e1e24; color: #ffdd57; border: 1px solid #2d2d30; border-radius: 4px; padding: 4px; font-size: 11px;")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setMinimumHeight(45)
        panel_layout.addWidget(self.lbl_warning)
        
        btn_box2 = QHBoxLayout()
        btn_reset = QPushButton("누수 상태 초기화")
        btn_reset.clicked.connect(self.reset_leaks)
        btn_free = QPushButton("즉시 안전 해제")
        btn_free.setObjectName("btnFreeNow")
        btn_free.clicked.connect(self.free_now)
        btn_box2.addWidget(btn_reset)
        btn_box2.addWidget(btn_free)
        panel_layout.addLayout(btn_box2)
        
        # 4. Output Options and Info
        panel_layout.addWidget(QLabel("4. 출력 옵션 및 정보"))
        info_layout = QHBoxLayout()
        self.cmb_sizemode = QComboBox()
        self.cmb_sizemode.addItems(["Normal", "StretchImage", "Zoom", "CenterImage"])
        self.cmb_sizemode.setCurrentIndex(1) # StretchImage
        info_layout.addWidget(self.cmb_sizemode)
        
        self.lbl_resolution = QLabel("해상도: -")
        self.lbl_resolution.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(self.lbl_resolution)
        
        self.lbl_channels = QLabel("채널 수: -")
        self.lbl_channels.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(self.lbl_channels)
        panel_layout.addLayout(info_layout)
        
        # 5. Brief summary
        txt_theory_summary = QTextEdit()
        txt_theory_summary.setReadOnly(True)
        txt_theory_summary.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 가비지 컬렉션과 참조 횟수\n"
            "   Python OpenCV는 ndarray를 사용하며, 참조 횟수가 0이 될 때 메모리가 자동으로 해제됩니다.\n"
            "2. 메모리 누수 재현 실험\n"
            "   이전 이미지의 참조를 끊지 않고(None 대입 누락) 계속해서 새 이미지를 로드하여 leaked_images 리스트에 추가하면 파이썬 가비지 컬렉터가 회수하지 못해 메모리가 누출됩니다.\n"
            "3. 안전 해제\n"
            "   src = None을 호출해 명시적으로 참조를 끊으면 원본 배열이 메모리에서 회수됩니다."
        )
        txt_theory_summary.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 4px; padding: 5px;")
        txt_theory_summary.setMaximumHeight(85)
        panel_layout.addWidget(txt_theory_summary)
        
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
            self.lbl_q1_result.setText("정답! 변수에 새로운 풍선 끈을 쥐여주면 이전 풍선은 끈이 끊겨 청소 로봇이 자동으로 수거해 가요.")
        elif self.rdo_q1_x.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q1_result.setText("오답입니다. 파이썬은 끈(참조)이 끊어지는 시점에 알아서 자동으로 수거하므로 별도의 수동 파괴 명령은 필요 없어요.")
        else:
            self.lbl_q1_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q1_result.setText("답안을 먼저 체크해 주세요.")
            
        # Q2 Check (Answer: O)
        if self.rdo_q2_o.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q2_result.setText("정답! 화면 표시는 단순히 복사본을 복제해서 붙여둔 것뿐이므로, 원본 변수 풍선은 내가 직접 None을 주입해 끈을 놓아야 비로소 사라져요.")
        elif self.rdo_q2_x.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q2_result.setText("오답입니다. 화면에 그려졌다고 해서 원본 numpy 이미지 배열 메모리가 저절로 소멸하는 것은 아니랍니다.")
        else:
            self.lbl_q2_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q2_result.setText("답안을 먼저 체크해 주세요.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "출력할 이미지 파일 선택", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.set_active_file(file_path)

    def load_default_image(self):
        default_path = self.get_resource_path("Italia.jpg")
        if default_path and os.path.exists(default_path):
            self.set_active_file(default_path)
        else:
            QMessageBox.warning(self, "파일 없음", "기본 이미지 'Italia.jpg'를 찾을 수 없습니다. 경로를 확인해 주세요.")

    def set_active_file(self, file_path):
        self.selected_file_path = file_path
        filename = os.path.basename(file_path)
        self.lbl_file_path.setText(f"선택됨: {filename} (대기 중)")
        self.lbl_warning.setText("상태: 파일 준비 완료. [1단계: 로드] 버튼을 눌러 메모리에 적재하세요.")
        self.lbl_warning.setStyleSheet("background-color: #1e1e24; color: #ffdd57; border: 1px solid #2d2d30; border-radius: 4px; padding: 4px; font-size: 11px;")
        
        self.btn_step1.setEnabled(True)
        self.btn_step2.setEnabled(False)
        self.btn_step3.setEnabled(False)

    def btn_step1_clicked(self):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            QMessageBox.warning(self, "알림", "파일을 먼저 선택해 주세요.")
            return
            
        try:
            # MEMORY LEAK SIMULATION:
            # In Python, if we assign to self.active_image without freeing the old one,
            # reference count drops and it frees automatically.
            # To simulate a C# leak, we append the unreleased self.active_image to self.leaked_images
            if self.active_image is not None:
                h, w = self.active_image.shape[:2]
                ch = self.active_image.shape[2] if len(self.active_image.shape) == 3 else 1
                leak_size = w * h * ch
                
                self.leaked_images.append(self.active_image) # Keep reference alive!
                self.total_leaked_bytes += leak_size
                
                self.lbl_warning.setText(f"[경고] 메모리 누출 발생! 이전 이미지({leak_size:,} Bytes)를 해제하지 않아 힙에 유실되었습니다.")
                self.lbl_warning.setStyleSheet("background-color: #3b1e22; color: #ef4565; border: 1px solid #632328; border-radius: 4px; padding: 4px; font-size: 11px;")
            else:
                self.lbl_warning.setText("상태: 1단계 완료. 이미지가 네이티브 힙에 로드되었습니다. (화면 미출력)")
                self.lbl_warning.setStyleSheet("background-color: #1a2430; color: #3e8ed0; border: 1px solid #233446; border-radius: 4px; padding: 4px; font-size: 11px;")
                
            # Support Unicode / Korean paths on Windows using np.fromfile and cv2.imdecode
            try:
                img_array = np.fromfile(self.selected_file_path, np.uint8)
                self.active_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                QMessageBox.warning(self, "오류", f"이미지 파일을 읽을 수 없습니다.\n{str(e)}")
                return
                
            if self.active_image is None:
                QMessageBox.warning(self, "오류", "이미지 디코딩에 실패했습니다.")
                return
                
            h, w = self.active_image.shape[:2]
            ch = self.active_image.shape[2] if len(self.active_image.shape) == 3 else 1
            self.active_allocated_bytes = w * h * ch
            
            # Update labels
            self.lbl_resolution.setText(f"해상도: {w} x {h}")
            self.lbl_channels.setText(f"채널 수: {ch} (BGR)")
            
            self.btn_step2.setEnabled(True)
            self.btn_step3.setEnabled(True)
            
            self.update_memory_gauges()
            
        except Exception as ex:
            QMessageBox.critical(self, "오류", f"이미지 로드 중 에러: {str(ex)}")

    def btn_step2_clicked(self):
        if self.active_image is None:
            QMessageBox.warning(self, "알림", "메모리에 로드된 이미지가 없습니다.")
            return
            
        disp = apply_size_mode(self.active_image, 640, 480, self.cmb_sizemode.currentText())
        pix = ndarray_to_qpixmap(disp)
        self.pic_box.setPixmap(pix)
        
        self.lbl_warning.setText("상태: 2단계 완료. 이미지가 PictureBox에 성공적으로 바인딩되었습니다.")
        self.lbl_warning.setStyleSheet("background-color: #1a2a24; color: #2cb67d; border: 1px solid #1a4233; border-radius: 4px; padding: 4px; font-size: 11px;")

    def btn_step3_clicked(self):
        self.free_image()

    def free_image(self):
        if self.active_image is not None:
            self.active_image = None
            self.active_allocated_bytes = 0
            self.pic_box.clear()
            
            self.lbl_warning.setText("상태: 3단계 완료. src = None을 통해 네이티브 리소스가 회수되었습니다.")
            self.lbl_warning.setStyleSheet("background-color: #1a2430; color: #3e8ed0; border: 1px solid #233446; border-radius: 4px; padding: 4px; font-size: 11px;")
            
            self.btn_step2.setEnabled(False)
            self.btn_step3.setEnabled(False)
            self.update_memory_gauges()

    def reset_leaks(self):
        self.leaked_images.clear()
        self.total_leaked_bytes = 0
        self.lbl_warning.setText("상태: 누수 메모리 기록을 초기화했습니다.")
        self.lbl_warning.setStyleSheet("background-color: #1e1e24; color: #ffdd57; border: 1px solid #2d2d30; border-radius: 4px; padding: 4px; font-size: 11px;")
        self.update_memory_gauges()

    def free_now(self):
        self.free_image()

    def update_memory_gauges(self):
        active_mb = self.active_allocated_bytes / (1024.0 * 1024.0)
        leaked_mb = self.total_leaked_bytes / (1024.0 * 1024.0)
        
        self.lbl_active_mem.setText(f"활성 네이티브 메모리: {self.active_allocated_bytes:,} Bytes ({active_mb:.2f} MB)")
        self.lbl_leaked_mem.setText(f"누출(누수) 메모리 합계: {self.total_leaked_bytes:,} Bytes ({leaked_mb:.2f} MB)")
        
        self.pb_active_mem.setValue(min(self.active_allocated_bytes, self.max_bytes))
        self.pb_leaked_mem.setValue(min(self.total_leaked_bytes, self.max_bytes))

    def closeEvent(self, event):
        self.active_image = None
        self.leaked_images.clear()
        super().closeEvent(event)
