import os
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QGroupBox, QRadioButton, QPushButton, QComboBox, 
    QFileDialog, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap

from ui_helpers import apply_size_mode, ndarray_to_qpixmap
from opencv_class import OpenCVClass

class FormCh06(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH06 - 클래스 설계 및 GrayScale 변환 실습")
        self.resize(1200, 740)
        self.setMinimumSize(1000, 700)
        
        # Class Instance and File State
        self.open_cv = None
        self.selected_file_path = ""
        
        self.init_ui()
        self.update_status()
        self.update_code_preview(0)
        
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
        
        lbl_title = QLabel("006. 클래스 설계 및 GrayScale 변환 핵심 이론 학습")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f5af0; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)
        
        txt_theory = QTextEdit()
        txt_theory.setReadOnly(True)
        txt_theory.setHtml(
            "<h3>■ [파이썬 자동 정리 상자: with 문과 자원 해제]</h3>"
            "<p>파이썬의 이미지(ndarray)는 가비지 컬렉터가 정리해 주지만, 대량의 사진을 한꺼번에 다루거나 외부 장치(파일, 카메라)를 연결할 때는 <b>'스스로 청소하는 상자'</b>를 만들어 두는 것이 안전합니다.</p>"
            "<ul>"
            "  <li><b>자동 정리 상자 (Context Manager: with문)</b>: <code>with</code> 블록을 만들어 놓으면, 블록을 빠져나갈 때 <code>__exit__</code> 메서드가 작동하여 내부 리소스들을 자동으로 정리(dispose)해 줍니다. C#의 <code>using</code> 블록과 똑같은 역할을 합니다.</li>"
            "  <li><b>수동 정리 (dispose)</b>: 필요할 때 <code>open_cv.dispose()</code>를 직접 호출하여 사용이 끝난 BGR 원본 이미지와 흑백 이미지의 끈을 끊어(None 대입) 힙 메모리를 깨끗하게 비우는 구조입니다.</li>"
            "</ul>"
            "<hr/>"
            "<h3>■ [끈 끊기(참조 해제)와 댕글링 방지]</h3>"
            "<p>파이썬에서 변수에 <code>None</code>을 넣는 것은 '빈 상자'로 만드는 것과 같습니다.</p>"
            "<ul>"
            "  <li><b>메모리 청소 활성화</b>: 더 이상 안 쓰는 이미지 ndarray를 붙잡고 있지 않도록 링크를 끊어 파이썬의 자동 청소(GC)를 돕습니다.</li>"
            "  <li><b>에러 예방</b>: 이미 청소한 변수에 실수로 또 접근하려고 할 때 즉각 에러(NoneType Error)를 내서 버그 위치를 쉽게 찾을 수 있게 도와주며, 메모리를 두 번 지우려는 실수를 막아줍니다.</li>"
            "</ul>"
            "<hr/>"
            "<h3>■ [컬러 BGR에서 Grayscale(흑백) 변환]</h3>"
            "<ul>"
            "  <li><b>그레이스케일</b>: 빨강, 초록, 파랑 색상 정보를 다 빼고, 오직 밝기(0~255)만 표현하는 1채널 격자 이미지입니다.</li>"
            "  <li><b>변환 가중치 공식</b>: 사람의 눈이 초록색에 가장 민감한 특징을 살려 공식에 따라 밝기를 구합니다.<br/>"
            "      &nbsp;&nbsp;&nbsp;&nbsp;<b>Gray = 0.299 * R + 0.587 * G + 0.114 * B</b></li>"
            "  <li><b>배낭 가볍게 만들기 (용량 절약)</b>: 색깔을 내는 3장의 색종이 겹수가 단 1장의 흑백 종이로 줄어들어, 용량이 <b>정확히 1/3</b>로 가벼워집니다. 덕분에 선 따기(에지 검출)나 얼굴 인식 같은 무거운 연산을 할 때 연산 속도를 엄청나게 빠르게 만들 수 있습니다.</li>"
            "</ul>"
        )
        txt_theory.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 10px;")
        left_layout.addWidget(txt_theory)
        
        # Quiz Area
        grp_quiz = QGroupBox("✍ 자가 진단 퀴즈 (이론 검증)")
        quiz_layout = QVBoxLayout(grp_quiz)
        
        # Quiz 1
        q1_group = QGroupBox("질문 1. 파이썬 자동 정리 상자 (with 문)")
        q1_layout = QVBoxLayout(q1_group)
        q1_lbl = QLabel("파이썬에서 `with` 구문을 사용해 블록을 벗어날 때 객체가 가진 리소스를 자동으로 해제(clean up)하는 기법은 C#의 `using` 블록과 유사한 개념이다.")
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
        q2_group = QGroupBox("질문 2. 흑백 이미지의 데이터 압축 효과")
        q2_layout = QVBoxLayout(q2_group)
        q2_lbl = QLabel("가로세로 크기가 같은 이미지를 BGR 컬러에서 흑백(Grayscale)으로 변환하면 색상 종이 겹수가 3장에서 1장으로 줄어들어, 데이터 용량이 정확히 3분의 1로 가벼워진다.")
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
        lbl_diag_title = QLabel("🖼 시각 자료: 클래스 설계 및 GrayScale 변환 파이프라인")
        lbl_diag_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f5af0;")
        right_layout.addWidget(lbl_diag_title)
        
        self.pic_diagram = QLabel()
        self.pic_diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic_diagram.setStyleSheet("background-color: #101012; border: 1px solid #2d2d30; border-radius: 6px;")
        
        img_path = self.get_resource_path("ch06_class_grayscale_ko.png")
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.pic_diagram.setPixmap(pix.scaled(580, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pic_diagram.setText("[이미지 파일 없음: ch06_class_grayscale_ko.png]")
            
        right_layout.addWidget(self.pic_diagram)
        
        txt_diag_desc = QTextEdit()
        txt_diag_desc.setReadOnly(True)
        txt_diag_desc.setPlainText(
            "【인포그래픽 설명】\n"
            "1. 커스텀 클래스 설계: OpenCVClass는 이미지 ndarray 자원들을 캡슐화하여 관리합니다.\n"
            "2. Grayscale 흑백 채널 변환: cv2.cvtColor(BGR2GRAY)를 적용하여 이미지 데이터를 단일 채널로 축소시킵니다. 변환 결과는 원본의 정확히 1/3 바이트 크기입니다.\n"
            "3. 파이썬 자원 해제: dispose() 메서드를 직접 실행하여 내부 BGR 및 Gray 이미지 ndarray 변수들의 참조를 끊어주면 파이썬의 참조 횟수가 0이 되면서 즉각 가비지 컬렉터에 의해 소멸 회수됩니다."
        )
        txt_diag_desc.setStyleSheet("background-color: #1a1a24; border: 1px solid #2d2d30; border-radius: 6px; padding: 5px;")
        txt_diag_desc.setMaximumHeight(150)
        right_layout.addWidget(txt_diag_desc)
        
        layout.addLayout(right_layout, 1)

    def init_lab_tab(self):
        layout = QHBoxLayout(self.tab_lab)
        
        # Left side: PictureBoxes (QLabel)
        left_layout = QVBoxLayout()
        pic_layout = QHBoxLayout()
        
        self.pb_original = QLabel()
        self.pb_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pb_original.setStyleSheet("background-color: #000000; border: 1px solid #2d2d30; border-radius: 6px;")
        self.pb_original.setFixedSize(350, 480)
        
        self.pb_gray = QLabel()
        self.pb_gray.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pb_gray.setStyleSheet("background-color: #000000; border: 1px solid #2d2d30; border-radius: 6px;")
        self.pb_gray.setFixedSize(350, 480)
        
        pic_layout.addWidget(self.pb_original)
        pic_layout.addWidget(self.pb_gray)
        left_layout.addLayout(pic_layout)
        
        txt_guide = QTextEdit()
        txt_guide.setReadOnly(True)
        txt_guide.setPlainText(
            "실습 가이드: 이미지를 지정한 뒤, 1단계부터 4단계까지 순서대로 실행하며 "
            "화면의 이미지 변화와 우측의 인스턴스 및 네이티브 메모리 상태의 동적 변화를 모니터링하세요."
        )
        txt_guide.setStyleSheet("background-color: #2b2b1a; color: #ffdd57; border: 1px solid #444433; border-radius: 4px; padding: 5px;")
        txt_guide.setMaximumHeight(50)
        left_layout.addWidget(txt_guide)
        layout.addLayout(left_layout)
        
        # Right side: Controls Panel
        panel = QGroupBox("006. 클래스 설계 & GrayScale 실습 제어")
        panel_layout = QVBoxLayout(panel)
        
        # 1. File Selection
        panel_layout.addWidget(QLabel("1. 이미지 파일 선택"))
        btn_box = QHBoxLayout()
        btn_open = QPushButton("파일 열기 (Open File)")
        btn_open.clicked.connect(self.select_file)
        btn_default = QPushButton("기본 이미지")
        btn_default.clicked.connect(self.load_default_image)
        btn_box.addWidget(btn_open)
        btn_box.addWidget(btn_default)
        panel_layout.addLayout(btn_box)
        
        self.lbl_file_path = QLabel("선택된 파일: 없음")
        self.lbl_file_path.setStyleSheet("color: #72727a; font-size: 11px;")
        self.lbl_file_path.setWordWrap(True)
        panel_layout.addWidget(self.lbl_file_path)
        
        # 2. Step-by-Step Interactive
        panel_layout.addWidget(QLabel("2. OpenCVClass 인스턴스 제어 단계"))
        
        self.btn_create = QPushButton("1단계: 클래스 객체 생성")
        self.btn_create.setObjectName("btnStart")  # green color from theme
        self.btn_create.clicked.connect(self.btn_create_clicked)
        panel_layout.addWidget(self.btn_create)
        
        self.btn_load_image = QPushButton("2단계: 이미지 로드 (Load)")
        self.btn_load_image.setObjectName("btnStep1")  # green color
        self.btn_load_image.clicked.connect(self.btn_load_image_clicked)
        panel_layout.addWidget(self.btn_load_image)
        
        self.btn_convert = QPushButton("3단계: Grayscale 변환")
        self.btn_convert.setObjectName("btnStep2")  # green color
        self.btn_convert.clicked.connect(self.btn_convert_clicked)
        panel_layout.addWidget(self.btn_convert)
        
        self.btn_dispose = QPushButton("4단계: 자원 해제 (Dispose)")
        self.btn_dispose.setObjectName("btnStop")  # red color
        self.btn_dispose.clicked.connect(self.btn_dispose_clicked)
        panel_layout.addWidget(self.btn_dispose)
        
        # 3. Memory Monitoring
        panel_layout.addWidget(QLabel("■ 인스턴스 및 메모리 상태"))
        self.lbl_class_status = QLabel("인스턴스 상태: 미생성 (Null)")
        self.lbl_class_status.setStyleSheet("font-weight: bold; color: #ef4565;")
        panel_layout.addWidget(self.lbl_class_status)
        
        self.lbl_image_status = QLabel("이미지 상태: 미로드")
        self.lbl_image_status.setStyleSheet("font-weight: bold; color: #ef4565;")
        panel_layout.addWidget(self.lbl_image_status)
        
        self.lbl_mem_info = QLabel("네이티브 메모리: 0 Bytes")
        self.lbl_mem_info.setStyleSheet("font-weight: bold; color: #7f5af0;")
        panel_layout.addWidget(self.lbl_mem_info)
        
        # 4. Code Preview
        panel_layout.addWidget(QLabel("3. 실행 코드 프리뷰 (Python)"))
        self.txt_code_preview = QTextEdit()
        self.txt_code_preview.setReadOnly(True)
        self.txt_code_preview.setStyleSheet("background-color: #1a1a24; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #2d2d30; border-radius: 4px; padding: 5px;")
        self.txt_code_preview.setMinimumHeight(110)
        panel_layout.addWidget(self.txt_code_preview)
        
        # 5. Brief summary
        txt_theory_summary = QTextEdit()
        txt_theory_summary.setReadOnly(True)
        txt_theory_summary.setPlainText(
            "【핵심 이론 및 원리】\n"
            "1. 클래스 캡슐화: 객체 내부에 이미지 자원을 관리하여 복잡한 로직 및 자원 관리를 단순화합니다.\n"
            "2. Grayscale 채널 축소: BGR 3채널을 1채널 밝기 데이터로 결합시켜 메모리 용량을 3분의 1로 압축합니다.\n"
            "3. 파이썬 자원 반납: 객체를 dispose하거나 None 대입 시 참조 횟수가 차감되면서 힙 영역이 즉각 반환됩니다."
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
            self.lbl_q1_result.setText("정답! `with` 문은 블록이 끝날 때 알아서 `__exit__` 메서드를 호출해 내부를 청소해 주는 편리한 자동 정리 도구예요.")
        elif self.rdo_q1_x.isChecked():
            self.lbl_q1_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q1_result.setText("오답입니다. 사용 후 리소스를 즉시 안전하게 수거하여 메모리 낭비를 줄이려는 개발 목적이 서로 정확히 부합해요.")
        else:
            self.lbl_q1_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q1_result.setText("답안을 먼저 체크해 주세요.")
            
        # Q2 Check (Answer: O)
        if self.rdo_q2_o.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #2cb67d; font-weight: bold;")
            self.lbl_q2_result.setText("정답! 빨강, 초록, 파랑 3장의 컬러 채널 색종이가 단 1장의 흑백 명암 종이로 합쳐지므로 정확히 3분의 1로 가벼워져요.")
        elif self.rdo_q2_x.isChecked():
            self.lbl_q2_result.setStyleSheet("color: #ef4565; font-weight: bold;")
            self.lbl_q2_result.setText("오답입니다. 해상도가 같을 때 3채널 컬러 데이터를 1채널 흑백 데이터로 바꾸면 데이터 크기가 정확히 3분의 1이 된답니다.")
        else:
            self.lbl_q2_result.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_q2_result.setText("답안을 먼저 체크해 주세요.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "로드할 이미지 선택", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file_path.setText(f"선택됨: {os.path.basename(file_path)}")
            self.update_status()

    def load_default_image(self):
        default_path = self.get_resource_path("Italia.jpg")
        if default_path and os.path.exists(default_path):
            self.selected_file_path = default_path
            self.lbl_file_path.setText(f"선택됨: {os.path.basename(default_path)}")
            self.update_status()
        else:
            QMessageBox.warning(self, "파일 없음", "기본 이미지 'Italia.jpg'를 찾을 수 없습니다. 경로를 확인해 주세요.")

    def btn_create_clicked(self):
        if self.open_cv is not None:
            return
        self.open_cv = OpenCVClass()
        self.update_status()
        self.update_code_preview(1)

    def btn_load_image_clicked(self):
        if self.open_cv is None:
            return
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            QMessageBox.warning(self, "알림", "이미지 파일을 먼저 선택해 주세요.")
            return

        try:
            self.open_cv.load_image(self.selected_file_path)
            
            # Display original image
            disp = apply_size_mode(self.open_cv.src_image, 350, 480, "Zoom")
            pix = ndarray_to_qpixmap(disp)
            self.pb_original.setPixmap(pix)
            
            # Clear gray image display (new load)
            self.pb_gray.clear()
            
            self.update_status()
            self.update_code_preview(2)
        except Exception as ex:
            QMessageBox.critical(self, "오류", f"이미지 로딩 중 실패: {str(ex)}")

    def btn_convert_clicked(self):
        if self.open_cv is None or self.open_cv.src_image is None:
            return

        try:
            self.open_cv.convert_to_gray()
            
            # Display grayscale image
            disp = apply_size_mode(self.open_cv.gray_image, 350, 480, "Zoom")
            pix = ndarray_to_qpixmap(disp)
            self.pb_gray.setPixmap(pix)
            
            self.update_status()
            self.update_code_preview(3)
        except Exception as ex:
            QMessageBox.critical(self, "오류", f"Grayscale 변환 중 실패: {str(ex)}")

    def btn_dispose_clicked(self):
        if self.open_cv is None:
            return

        self.open_cv.dispose()
        self.open_cv = None

        self.pb_original.clear()
        self.pb_gray.clear()

        self.update_status()
        self.update_code_preview(4)

    def update_status(self):
        if self.open_cv is None:
            self.lbl_class_status.setText("인스턴스 상태: 미생성 (Null)")
            self.lbl_class_status.setStyleSheet("font-weight: bold; color: #ef4565;")
            self.lbl_image_status.setText("이미지 상태: 미로드")
            self.lbl_image_status.setStyleSheet("font-weight: bold; color: #ef4565;")
            self.lbl_mem_info.setText("네이티브 메모리: 0 Bytes")
            
            self.btn_create.setEnabled(True)
            self.btn_load_image.setEnabled(False)
            self.btn_convert.setEnabled(False)
            self.btn_dispose.setEnabled(False)
        else:
            self.lbl_class_status.setText("인스턴스 상태: 생성됨 (Active)")
            self.lbl_class_status.setStyleSheet("font-weight: bold; color: #2cb67d;")
            
            total_bytes = 0
            img_status = "미로드"
            self.lbl_image_status.setStyleSheet("font-weight: bold; color: #ef4565;")
            
            if self.open_cv.src_image is not None:
                h, w = self.open_cv.src_image.shape[:2]
                ch = self.open_cv.src_image.shape[2] if len(self.open_cv.src_image.shape) == 3 else 1
                total_bytes += w * h * ch
                img_status = f"원본 로드됨 ({w}x{h})"
                self.lbl_image_status.setStyleSheet("font-weight: bold; color: #e67e22;")

            if self.open_cv.gray_image is not None:
                h, w = self.open_cv.gray_image.shape[:2]
                ch = self.open_cv.gray_image.shape[2] if len(self.open_cv.gray_image.shape) == 3 else 1
                total_bytes += w * h * ch
                img_status = "Grayscale 변환 완료"
                self.lbl_image_status.setStyleSheet("font-weight: bold; color: #2cb67d;")

            self.lbl_image_status.setText(f"이미지 상태: {img_status}")
            self.lbl_mem_info.setText(f"네이티브 메모리: {total_bytes:,} Bytes ({total_bytes / 1024.0 / 1024.0:.2f} MB)")
            
            self.btn_create.setEnabled(False)
            self.btn_load_image.setEnabled(bool(self.selected_file_path))
            self.btn_convert.setEnabled(self.open_cv.src_image is not None)
            self.btn_dispose.setEnabled(True)

    def update_code_preview(self, step):
        self.txt_code_preview.clear()
        
        filename = os.path.basename(self.selected_file_path) if self.selected_file_path else "Italia.jpg"
        
        code = ""
        if step == 0:
            code = (
                "# OpenCVClass 인스턴스 미생성\n"
                "open_cv = None"
            )
        elif step == 1:
            code = (
                "# 1단계: 클래스 인스턴스 생성\n"
                "open_cv = OpenCVClass()\n"
                "# (인스턴스가 메모리에 적재됨)"
            )
        elif step == 2:
            code = (
                "# 2단계: 이미지 로드\n"
                f"# file_path = \"{filename}\"\n"
                "open_cv.load_image(file_path)\n"
                "# (내부 _src_image ndarray 생성 완료)"
            )
        elif step == 3:
            code = (
                "# 3단계: Grayscale 변환 실행\n"
                "open_cv.convert_to_gray()\n"
                "# (내부 _gray_image ndarray 변환 생성)"
            )
        elif step == 4:
            code = (
                "# 4단계: dispose 호출하여 리소스 수동 해제\n"
                "open_cv.dispose()\n"
                "open_cv = None\n"
                "# (NumPy 배열 참조가 소멸하며 GC가 메모리 회수)"
            )
            
        self.txt_code_preview.setText(code)

    def closeEvent(self, event):
        if self.open_cv is not None:
            self.open_cv.dispose()
            self.open_cv = None
        super().closeEvent(event)
