import os
import sys
from PIL import Image, ImageDraw, ImageFont

def draw_ch02_diagram(output_path, font_path):
    # Width: 1160, Height: 960 (for high res, scaled down dynamically by QLabel to 580x480)
    w, h = 1160, 960
    img = Image.new("RGB", (w, h), "#121214")
    draw = ImageDraw.Draw(img)
    
    # Fonts
    try:
        font_title = ImageFont.truetype(font_path, 40)
        font_subtitle = ImageFont.truetype(font_path, 28)
        font_text = ImageFont.truetype(font_path, 22)
        font_small = ImageFont.truetype(font_path, 18)
    except:
        font_title = font_subtitle = font_text = font_small = ImageFont.load_default()
        
    # Title
    draw.text((w // 2, 50), "그레이스케일 vs BGR 컬러 이미지 구조 (Python np.ndarray)", fill="#ffffff", font=font_title, anchor="mm")
    
    # 1. Grayscale Card
    # x: 80 to 520 (width 440)
    draw.rounded_rectangle([80, 120, 520, 880], radius=15, fill="#1e1e24", outline="#2d2d30", width=2)
    draw.text((300, 160), "1. 그레이스케일 (명암 이미지)", fill="#2cb67d", font=font_subtitle, anchor="mm")
    draw.text((300, 210), "형태 (Shape): (H, W)\n자료형 (Dtype): np.uint8", fill="#a0a0a5", font=font_text, anchor="mm")
    
    # Draw Grayscale Grid
    # Draw 4x4 grid to represent pixels
    grid_size = 60
    start_x, start_y = 180, 280
    for i in range(4):
        for j in range(4):
            val = (i + j) * 40
            bx = start_x + j * grid_size
            by = start_y + i * grid_size
            draw.rectangle([bx, by, bx+grid_size, by+grid_size], fill=(val, val, val), outline="#3f3f46")
            draw.text((bx+grid_size//2, by+grid_size//2), str(val), fill="#ffffff" if val < 128 else "#000000", font=font_small, anchor="mm")
            
    draw.text((300, 560), "■ 특징\n- 픽셀당 1바이트 (0~255 밝기 값) 소모\n- 연산량 최소, 전처리 1순위로 사용\n\n■ 이론적 메모리 용량 계산\n용량 = 가로(W) x 세로(H) x 1 Byte\n예) 640 x 480 x 1 = 307,200 Bytes\n(약 300 KB)", fill="#e3e3e6", font=font_text, anchor="lm")
    
    # 2. BGR Color Card
    # x: 640 to 1080 (width 440)
    draw.rounded_rectangle([640, 120, 1080, 880], radius=15, fill="#1e1e24", outline="#2d2d30", width=2)
    draw.text((860, 160), "2. BGR 컬러 (색상 이미지)", fill="#7f5af0", font=font_subtitle, anchor="mm")
    draw.text((860, 210), "형태 (Shape): (H, W, 3)\n자료형 (Dtype): np.uint8", fill="#a0a0a5", font=font_text, anchor="mm")
    
    # Draw 3D-like BGR stacked channels
    # Red channel
    start_x, start_y = 780, 360
    offset = 25
    
    # Red Channel (bottom)
    draw.rectangle([start_x, start_y, start_x+160, start_y+120], fill=(180, 20, 20), outline="#ef4565", width=2)
    draw.text((start_x+80, start_y+60), "R 채널\n(1 Byte)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Green Channel (middle)
    draw.rectangle([start_x-offset, start_y-offset, start_x+160-offset, start_y+120-offset], fill=(20, 150, 20), outline="#2cb67d", width=2)
    draw.text((start_x+80-offset, start_y+60-offset), "G 채널\n(1 Byte)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Blue Channel (top)
    draw.rectangle([start_x-offset*2, start_y-offset*2, start_x+160-offset*2, start_y+120-offset*2], fill=(20, 20, 180), outline="#3e8ed0", width=2)
    draw.text((start_x+80-offset*2, start_y+60-offset*2), "B 채널\n(1 Byte)", fill="#ffffff", font=font_text, anchor="mm")
    
    draw.text((860, 560), "■ 특징\n- 픽셀당 3바이트 (Blue, Green, Red) 소모\n- 인간이 인지 가능한 다채로운 색 구현\n- 그레이스케일 대비 정확히 3배 무거움\n\n■ 이론적 메모리 용량 계산\n용량 = 가로(W) x 세로(H) x 3 Bytes\n예) 640 x 480 x 3 = 921,600 Bytes\n(약 900 KB)", fill="#e3e3e6", font=font_text, anchor="lm")
    
    # Footer Note
    draw.text((w // 2, 915), "결론: 해상도와 채널수가 증가하면 하드웨어 대역폭/메모리 부하가 급증합니다.", fill="#ffdd57", font=font_text, anchor="mm")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved: {output_path}")

def draw_ch03_diagram(output_path, font_path):
    w, h = 1160, 960
    img = Image.new("RGB", (w, h), "#121214")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype(font_path, 40)
        font_subtitle = ImageFont.truetype(font_path, 28)
        font_text = ImageFont.truetype(font_path, 22)
    except:
        font_title = font_subtitle = font_text = ImageFont.load_default()
        
    # Title
    draw.text((w // 2, 50), "실시간 웹캠 영상 획득 및 필터 처리 파이프라인 (Python)", fill="#ffffff", font=font_title, anchor="mm")
    
    # Pipeline stages
    stages = [
        {"title": "1. 웹캠 하드웨어", "desc": "아날로그 빛 캡처\n& 디지털 비디오 방출", "color": "#72727a"},
        {"title": "2. cv2.VideoCapture", "desc": "웹캠 자원 연결\ncap = cv2.VideoCapture(0)\n프레임 획득: ret, frame = cap.read()", "color": "#7f5af0"},
        {"title": "3. OpenCV 이미지 연산", "desc": "Grayscale: cv2.cvtColor(BGR2GRAY)\nCanny 에지: cv2.Canny(gray, 50, 150)\n(NumPy ndarray 행렬 기반)", "color": "#ef4565"},
        {"title": "4. PyQt6 GUI 화면 출력", "desc": "QImage/QPixmap으로 픽셀 복사\nQLabel.setPixmap(pixmap)\n(GUI 메인 스레드 렌더링)", "color": "#2cb67d"}
    ]
    
    y = 140
    box_w = 900
    box_h = 150
    start_x = 130
    
    for i, stage in enumerate(stages):
        # Draw Box
        draw.rounded_rectangle([start_x, y, start_x+box_w, y+box_h], radius=10, fill="#1e1e24", outline=stage["color"], width=2)
        
        # Draw text inside
        draw.text((start_x + 30, y + box_h // 2), stage["title"], fill=stage["color"], font=font_subtitle, anchor="lm")
        draw.text((start_x + 350, y + box_h // 2), stage["desc"], fill="#e3e3e6", font=font_text, anchor="lm")
        
        # Draw Arrow to next stage
        if i < len(stages) - 1:
            ay = y + box_h
            draw.line([w // 2, ay, w // 2, ay + 35], fill="#ffdd57", width=3)
            # Arrow head
            draw.polygon([w // 2 - 10, ay + 25, w // 2 + 10, ay + 25, w // 2, ay + 35], fill="#ffdd57")
            
        y += box_h + 35
        
    # Footer warning
    draw.text((w // 2, 915), "주의: cap.release()를 호출하지 않으면 웹캠 장치가 잠겨 다른 프로그램에서 켜지지 않습니다.", fill="#ffdd57", font=font_text, anchor="mm")
    
    img.save(output_path)
    print(f"Saved: {output_path}")

def draw_ch04_diagram(output_path, font_path):
    w, h = 1160, 960
    img = Image.new("RGB", (w, h), "#121214")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype(font_path, 40)
        font_subtitle = ImageFont.truetype(font_path, 28)
        font_text = ImageFont.truetype(font_path, 22)
    except:
        font_title = font_subtitle = font_text = ImageFont.load_default()
        
    # Title
    draw.text((w // 2, 50), "동영상 파일 디코딩(재생) vs 인코딩(녹화) 생명 주기", fill="#ffffff", font=font_title, anchor="mm")
    
    # 1. Decoding Card (Left)
    draw.rounded_rectangle([80, 120, 550, 880], radius=15, fill="#1e1e24", outline="#3e8ed0", width=2)
    draw.text((315, 160), "1. 동영상 재생 (디코딩)", fill="#3e8ed0", font=font_subtitle, anchor="mm")
    
    dec_steps = [
        "① 파일 초기화\n   cap = cv2.VideoCapture(file_path)\n   (동영상 컨테이너 열기 및 메타데이터 획득)",
        "② 프레임 루프 (Timer)\n   ret, frame = cap.read()\n   (압축된 패킷 디코딩 후 NumPy 배열 획득)",
        "③ 위치 이동 (Seek)\n   cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n   (탐색 바 조작 시 해당 프레임으로 이동)",
        "④ 자원 해제\n   cap.release()\n   (동영상 파일 닫기 및 디코더 메모리 해제)"
    ]
    dy = 220
    for step in dec_steps:
        draw.text((110, dy), step, fill="#e3e3e6", font=font_text, anchor="lm")
        dy += 150
        
    # 2. Encoding Card (Right)
    draw.rounded_rectangle([610, 120, 1080, 880], radius=15, fill="#1e1e24", outline="#ef4565", width=2)
    draw.text((845, 160), "2. 동영상 저장 (인코딩)", fill="#ef4565", font=font_subtitle, anchor="mm")
    
    enc_steps = [
        "① 인코더 초기화\n   writer = cv2.VideoWriter(\n      path, fourcc, fps, (width, height)\n   )\n   (XVID, MJPG 등 코덱 지정)",
        "② 프레임 캡처 & 가공\n   ret, frame = cap.read()\n   (카메라 이미지 획득 후 리사이즈 수행)",
        "③ 스트림 주입\n   writer.write(frame_resized)\n   (프레임을 압축 코덱 파일로 순차 기록)",
        "④ 파일 닫기 (필수!)\n   writer.release()\n   (헤더 및 인덱스 최종 기록. 누락 시 0KB 파일)"
    ]
    ey = 220
    for step in enc_steps:
        draw.text((640, ey), step, fill="#e3e3e6", font=font_text, anchor="lm")
        ey += 150
        
    draw.text((w // 2, 915), "핵심: 인코딩 작업 후 release()를 하지 않으면 비디오 컨테이너 헤더 정보가 유실되어 재생 불가능한 파일이 됩니다.", fill="#ffdd57", font=font_text, anchor="mm")
    
    img.save(output_path)
    print(f"Saved: {output_path}")

def draw_ch05_diagram(output_path, font_path):
    w, h = 1160, 960
    img = Image.new("RGB", (w, h), "#121214")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype(font_path, 40)
        font_subtitle = ImageFont.truetype(font_path, 28)
        font_text = ImageFont.truetype(font_path, 20)
        font_large_text = ImageFont.truetype(font_path, 24)
    except:
        font_title = font_subtitle = font_text = font_large_text = ImageFont.load_default()
        
    # Title
    draw.text((w // 2, 50), "Python 가비지 컬렉션(참조 카운팅) & 메모리 누수 원리", fill="#ffffff", font=font_title, anchor="mm")
    
    # 1. Normal Memory Release Card (Left)
    draw.rounded_rectangle([80, 120, 550, 880], radius=15, fill="#1e1e24", outline="#2cb67d", width=2)
    draw.text((315, 160), "1. 정상적인 메모리 해제 흐름", fill="#2cb67d", font=font_subtitle, anchor="mm")
    
    # Diagram 1: Variable -> Object (Ref Count: 1)
    draw.rectangle([130, 240, 240, 300], fill="#7f5af0", outline="#ffffff")
    draw.text((185, 270), "src 변수", fill="#ffffff", font=font_text, anchor="mm")
    draw.line([240, 270, 340, 270], fill="#2cb67d", width=3)
    draw.polygon([330, 265, 330, 275, 340, 270], fill="#2cb67d")
    
    draw.rectangle([340, 230, 500, 310], fill="#121214", outline="#2cb67d", width=2)
    draw.text((420, 270), "ndarray 객체\n(Ref Count: 1)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Diagram 1 Action: src = None
    draw.text((315, 360), "[참조 끊기 실행]\nsrc = None", fill="#2cb67d", font=font_large_text, anchor="mm")
    
    # Diagram 1 State 2: No pointer -> Object (Ref Count: 0) -> GC Reclaims
    draw.rectangle([130, 460, 240, 520], fill="#72727a", outline="#ffffff")
    draw.text((185, 490), "src (None)", fill="#ffffff", font=font_text, anchor="mm")
    
    draw.rectangle([340, 450, 500, 530], fill="#121214", outline="#ef4565", width=2)
    draw.text((420, 490), "ndarray 객체\n(Ref Count: 0)", fill="#e3e3e6", font=font_text, anchor="mm")
    
    draw.line([420, 530, 420, 580], fill="#ef4565", width=3)
    draw.polygon([415, 570, 425, 570, 420, 580], fill="#ef4565")
    draw.rounded_rectangle([320, 580, 520, 640], radius=5, fill="#ef4565")
    draw.text((420, 610), "파이썬 GC가 즉시 해제", fill="#ffffff", font=font_text, anchor="mm")
    
    draw.text((315, 760), "■ 설명\n- 변수에 None을 대입하거나 함수가 종료되어\n  변수가 사라지면 참조 카운트가 0이 됩니다.\n- 참조 카운트가 0인 객체는 파이썬 가비지\n  컬렉터가 메모리를 즉시 완벽하게 회수합니다.", fill="#e3e3e6", font=font_text, anchor="mm")

    # 2. Memory Leak Card (Right)
    draw.rounded_rectangle([610, 120, 1080, 880], radius=15, fill="#1e1e24", outline="#ef4565", width=2)
    draw.text((845, 160), "2. 메모리 누수(유실) 발생 흐름", fill="#ef4565", font=font_subtitle, anchor="mm")
    
    # Diagram 2: src -> Object, leaked_list -> Object (Ref Count: 2)
    draw.rectangle([660, 230, 770, 280], fill="#7f5af0", outline="#ffffff")
    draw.text((715, 255), "src 변수", fill="#ffffff", font=font_text, anchor="mm")
    draw.line([770, 255, 870, 255], fill="#2cb67d", width=3)
    draw.polygon([860, 250, 860, 260, 870, 255], fill="#2cb67d")
    
    draw.rectangle([660, 290, 770, 340], fill="#7f5af0", outline="#ffffff")
    draw.text((715, 315), "leaked_list", fill="#ffffff", font=font_text, anchor="mm")
    draw.line([770, 315, 870, 315], fill="#2cb67d", width=3)
    draw.polygon([860, 310, 860, 320, 870, 315], fill="#2cb67d")
    
    draw.rectangle([870, 240, 1030, 330], fill="#121214", outline="#ef4565", width=2)
    draw.text((950, 285), "ndarray 객체\n(Ref Count: 2)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Diagram 2 Action: src = new_image (Reassignment)
    draw.text((845, 380), "[새 이미지 재대입 실행]\nsrc = cv2.imread(new_path)", fill="#ef4565", font=font_large_text, anchor="mm")
    
    # Diagram 2 State 2: src -> New Object, leaked_list -> Old Object (Ref Count: 1)
    draw.rectangle([660, 450, 770, 500], fill="#7f5af0", outline="#ffffff")
    draw.text((715, 475), "src 변수", fill="#ffffff", font=font_text, anchor="mm")
    draw.line([770, 475, 870, 475], fill="#2cb67d", width=3)
    draw.polygon([860, 470, 860, 480, 870, 475], fill="#2cb67d")
    
    draw.rectangle([870, 440, 1030, 510], fill="#121214", outline="#2cb67d", width=2)
    draw.text((950, 475), "새 ndarray 객체\n(Ref Count: 1)", fill="#ffffff", font=font_text, anchor="mm")
    
    draw.rectangle([660, 530, 770, 580], fill="#7f5af0", outline="#ffffff")
    draw.text((715, 555), "leaked_list", fill="#ffffff", font=font_text, anchor="mm")
    draw.line([770, 555, 870, 555], fill="#ef4565", width=3)
    draw.polygon([860, 550, 860, 560, 870, 555], fill="#ef4565")
    
    draw.rectangle([870, 520, 1030, 590], fill="#121214", outline="#ef4565", width=2)
    draw.text((950, 555), "이전 ndarray 객체\n(Ref Count: 1)", fill="#a0a0a5", font=font_text, anchor="mm")
    
    draw.rounded_rectangle([830, 610, 1070, 670], radius=5, fill="#ef4565")
    draw.text((950, 640), "참조가 해제되지 않아 누출 유지!", fill="#ffffff", font=font_text, anchor="mm")
    
    draw.text((845, 760), "■ 설명\n- 이전 이미지의 참조를 끊지 않고 다른 참조가\n  살아있는 상태에서 새 이미지를 할당하면,\n  이전 이미지의 참조 카운트는 0이 되지 않습니다.\n- 해제되지 않고 남아있는 무거운 이미지 데이터들이\n  누적되어 컴퓨터 메모리를 고갈시킵니다.", fill="#e3e3e6", font=font_text, anchor="mm")
    
    draw.text((w // 2, 915), "결론: 불필요한 이미지 객체는 리스트 등에서 삭제하고 변수를 None으로 할당하여 참조 카운트를 0으로 만들어 줘야 합니다.", fill="#ffdd57", font=font_text, anchor="mm")
    
    img.save(output_path)
    print(f"Saved: {output_path}")

def draw_ch06_diagram(output_path, font_path):
    w, h = 1160, 960
    img = Image.new("RGB", (w, h), "#121214")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype(font_path, 40)
        font_subtitle = ImageFont.truetype(font_path, 28)
        font_text = ImageFont.truetype(font_path, 20)
        font_large_text = ImageFont.truetype(font_path, 24)
    except:
        font_title = font_subtitle = font_text = font_large_text = ImageFont.load_default()
        
    # Title
    draw.text((w // 2, 50), "OpenCVClass 커스텀 설계 및 Grayscale 변환 메커니즘", fill="#ffffff", font=font_title, anchor="mm")
    
    # 1. Custom Class Card (Left)
    draw.rounded_rectangle([80, 120, 550, 880], radius=15, fill="#1e1e24", outline="#7f5af0", width=2)
    draw.text((315, 160), "1. OpenCVClass 자원 캡슐화 설계", fill="#7f5af0", font=font_subtitle, anchor="mm")
    
    class_steps = [
        "① 인스턴스 초기화 (__init__)\n   self._src_image = None\n   self._gray_image = None\n   (이미지 객체를 가리킬 내부 참조 변수 준비)",
        "② 이미지 로드 (load_image)\n   self._src_image = cv2.imdecode(...)\n   (한글 경로 우회 바이너리 디코딩 수행)",
        "③ 흑백 변환 (convert_to_gray)\n   self._gray_image = cv2.cvtColor(...)\n   (BGR 컬러에서 Grayscale 채널로 계산 수행)",
        "④ 수동 자원 반납 (dispose)\n   self._src_image = None\n   self._gray_image = None\n   (참조 해제로 GC가 ndarray 즉시 해제하도록 유도)"
    ]
    dy = 220
    for step in class_steps:
        draw.text((110, dy), step, fill="#e3e3e6", font=font_text, anchor="lm")
        dy += 150
        
    # 2. BGR vs GrayScale Conversion Card (Right)
    draw.rounded_rectangle([610, 120, 1080, 880], radius=15, fill="#1e1e24", outline="#2cb67d", width=2)
    draw.text((845, 160), "2. BGR 컬러 → Grayscale 변환 원리", fill="#2cb67d", font=font_subtitle, anchor="mm")
    
    # Draw stacked BGR grid
    draw.rounded_rectangle([660, 230, 800, 330], radius=5, fill="#3b1e22", outline="#ef4565", width=2)
    draw.text((730, 280), "BGR 컬러\n(3개 채널)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Draw arrow in between
    draw.line([830, 280, 910, 280], fill="#ffdd57", width=3)
    draw.polygon([900, 275, 900, 285, 910, 280], fill="#ffdd57")
    
    # Draw Gray grid
    draw.rounded_rectangle([940, 230, 1050, 330], radius=5, fill="#1a2430", outline="#3e8ed0", width=2)
    draw.text((995, 280), "Grayscale\n(1개 채널)", fill="#ffffff", font=font_text, anchor="mm")
    
    # Add explanations
    draw.text((640, 380), "■ 변환 공식 (밝기 인지 가중치 합)\n  Gray = 0.299 * R + 0.587 * G + 0.114 * B\n  (인간의 눈이 초록색에 가장 민감한 점 반영)", fill="#e3e3e6", font=font_text, anchor="lm")
    
    draw.text((640, 520), "■ 데이터 압축 효과 (1/3 축소)\n  - BGR: 픽셀당 3 Bytes 소모 (Blue, Green, Red)\n  - Gray: 픽셀당 1 Byte 소모 (밝기 밝기값만 존재)\n  - 이미지 해상도가 동일할 때 메모리 크기가\n    정확히 3분의 1로 압축되어 전송/연산 속도 향상", fill="#e3e3e6", font=font_text, anchor="lm")
    
    draw.text((640, 710), "■ 주요 활용 목적\n  - 에지 검출 (Canny Edge Detection)\n  - 형상 인식 (Template Matching, OCR 등)\n  - 굳이 색상 정보가 필요 없는 모든 기하학적 분석", fill="#e3e3e6", font=font_text, anchor="lm")
    
    draw.text((w // 2, 915), "결론: OpenCVClass 클래스를 활용하여 파이썬의 자원 생명 주기를 올바르게 캡슐화하고, 흑백 변환을 통해 연산 대역폭을 극대화합니다.", fill="#ffdd57", font=font_text, anchor="mm")
    
    img.save(output_path)
    print(f"Saved: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resources_dir = os.path.join(base_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    
    # Font path
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if not os.path.exists(font_path):
        # Fallback to malgunbd.ttf or default
        font_path = "C:/Windows/Fonts/malgunbd.ttf"
        
    draw_ch02_diagram(os.path.join(resources_dir, "ch02_pixel_structure_ko.png"), font_path)
    draw_ch03_diagram(os.path.join(resources_dir, "ch03_camera_pipeline_ko.png"), font_path)
    draw_ch04_diagram(os.path.join(resources_dir, "ch04_video_lifecycle_ko.png"), font_path)
    draw_ch05_diagram(os.path.join(resources_dir, "ch05_memory_leak_ko.png"), font_path)
    draw_ch06_diagram(os.path.join(resources_dir, "ch06_class_grayscale_ko.png"), font_path)
    print("All diagrams regenerated successfully for Python OpenCV!")

if __name__ == "__main__":
    main()
