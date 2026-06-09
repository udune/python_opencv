import cv2
import numpy as np

# 1-1. 설정 (원본 이미지 크기: 900x400, 표시 영역: 400x300)
W, H = 900, 400
TW, TH = 400, 300
mode = "Normal"

# 테스트용 간단한 원본 이미지 생성 (중앙에 원이 그려진 파란 이미지)
orig = np.zeros((H, W, 3), dtype=np.uint8)
orig[:] = (255, 100, 0) # 파란색 배경
# [OpenCV 핵심] 도형(원) 및 텍스트 그리기 API
cv2.circle(orig, (W // 2, H // 2), 80, (0, 255, 255), -1) # 중앙 노란 원
cv2.putText(orig, "Original 900x400", (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 1-3. SizeMode 4종 구현
def get_display_img(img, target_w, target_h, mode):
    h, w = img.shape[:2]
    if mode == "Normal":
        # numpy 슬라이싱
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:min(h, target_h), :min(w, target_w)] = img[:min(h, target_h), :min(w, target_w)]
        return canvas
    
    elif mode == "CenterImage":
        # 중앙 기준 numpy 슬라이싱
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        src_x = max((w - target_w) // 2, 0)
        src_y = max((h - target_h) // 2, 0)
        copy_w = min(w, target_w)
        copy_h = min(h, target_h)
        canvas[:copy_h, :copy_w] = img[src_y:src_y+copy_h, src_x:src_x+copy_w]
        return canvas
        
    elif mode == "StretchImage":
        # [OpenCV 핵심] cv2.resize를 활용한 강제 리사이즈 (비율 깨짐)
        return cv2.resize(img, (target_w, target_h))
        
    elif mode == "Zoom":
        # [OpenCV 핵심] 비율 유지 리사이즈 후 copyMakeBorder를 사용하여 레터박스(경계선 패딩) 구현
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        
        # 패딩 크기 계산
        pad_w, pad_h = target_w - nw, target_h - nh
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# 1-4. 좌표 역변환
def to_original_coords(disp_x, disp_y, mode):
    if mode == "Normal":
        return disp_x, disp_y if (0 <= disp_x < W and 0 <= disp_y < H) else None
        
    elif mode == "CenterImage":
        src_x = max((w - target_w) // 2, 0) if 'w' in globals() else max((W - TW) // 2, 0)
        src_y = max((h - target_h) // 2, 0) if 'h' in globals() else max((H - TH) // 2, 0)
        orig_x = disp_x + src_x
        orig_y = disp_y + src_y
        return (orig_x, orig_y) if (0 <= orig_x < W and 0 <= orig_y < H) else None
        
    elif mode == "StretchImage":
        return int(disp_x * (W / TW)), int(disp_y * (H / TH))
        
    elif mode == "Zoom":
        scale = min(TW / W, TH / H)
        nw, nh = int(W * scale), int(H * scale)
        left_pad = (TW - nw) // 2
        top_pad = (TH - nh) // 2
        if left_pad <= disp_x < left_pad + nw and top_pad <= disp_y < top_pad + nh:
            return int((disp_x - left_pad) / scale), int((disp_y - top_pad) / scale)
    return None

# 마우스 콜백 함수
def click_event(event, x, y, flags, param):
    # [OpenCV 핵심] 마우스 왼쪽 버튼 클릭 이벤트 감지
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_pos = to_original_coords(x, y, mode)
        print(f"[클릭] 화면: ({x}, {y}) -> 원본 역변환: {orig_pos}")
        
        if orig_pos:
            # 원본 이미지 창에 마킹
            marked = orig.copy()
            # [OpenCV 핵심] 매핑된 원본 좌표에 빨간 점을 찍어 출력
            cv2.circle(marked, orig_pos, 10, (0, 0, 255), -1)
            cv2.imshow("Original", marked)

# [OpenCV 핵심] 윈도우 생성 및 마우스 콜백 함수 연결
cv2.namedWindow("Display")
cv2.setMouseCallback("Display", click_event)

print("키 설정: 1 (Normal), 2 (Center), 3 (Stretch), 4 (Zoom), q (종료)")

while True:
    disp = get_display_img(orig, TW, TH, mode)
    cv2.putText(disp, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # [OpenCV 핵심] 화면 출력(imshow) 및 사용자 키 입력 대기(waitKey)
    cv2.imshow("Display", disp)
    cv2.imshow("Original", orig)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'): mode = "Normal"
    elif key == ord('2'): mode = "CenterImage"
    elif key == ord('3'): mode = "StretchImage"
    elif key == ord('4'): mode = "Zoom"

cv2.destroyAllWindows()
