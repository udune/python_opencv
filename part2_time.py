import cv2
import numpy as np
import time

# 2-1. 설정 (목표 FPS 및 프레임 주기 계산)
TARGET_FPS = 30
INTERVAL = 1.0 / TARGET_FPS  # 1프레임당 목표 시간 (약 0.033초)

mode = "Precision"  # "Simple" (고정 지연) vs "Precision" (동적 지연)
workload = False    # 가상 연산 부하 (ON/OFF)

# 가상 비디오 프레임 생성기 (좌우로 움직이는 원)
def get_frame(frame_cnt):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    x = int(320 + 200 * np.sin(frame_cnt * 0.1))
    # [OpenCV 핵심] 프레임 가상 렌더링에 원 그리기 사용
    cv2.circle(img, (x, 240), 30, (0, 255, 255), -1)
    return img

print("조작키: 1 (Simple Mode), 2 (Precision Mode), b (부하 토글), q (종료)")

frame_cnt = 0
prev_time = time.perf_counter()

while True:
    start_time = time.perf_counter()
    
    # 1. 프레임 획득
    frame = get_frame(frame_cnt)
    frame_cnt += 1
    
    # 2. 가상 CPU 연산 부하 (15ms 임의 지연 추가)
    if workload:
        time.sleep(0.015)
        
    # 3. 실시간 실제 FPS 측정
    curr_time = time.perf_counter()
    elapsed_frame = curr_time - prev_time
    prev_time = curr_time
    real_fps = 1.0 / elapsed_frame if elapsed_frame > 0 else 0
    
    # HUD 정보 출력
    cv2.putText(frame, f"Mode: {mode} (Press 1 or 2)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Workload: {workload} (Press B)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if workload else (0, 255, 0), 2)
    cv2.putText(frame, f"Real FPS: {real_fps:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    # [OpenCV 핵심] imshow를 이용해 실시간 렌더링 화면 출력
    cv2.imshow("Time Control Lab", frame)
    
    # 4. 루프 소요 시간 계산
    loop_time = time.perf_counter() - start_time
    
    # 5. 모드별 waitKey 지연 계산
    # [OpenCV 핵심] waitKey 대기 지연 적용 (정밀 보정된 wait_ms 또는 고정 지연 적용)
    # 2-3 & 2-4의 대기 시간(delay) 처리를 이 waitKey가 수행함
    if mode == "Precision":
        # 정밀 보정: 목표 주기에서 프레임 처리 시간을 뺀 만큼만 대기
        wait_ms = int((INTERVAL - loop_time) * 1000)
        wait_ms = max(wait_ms, 1)
    else:
        # 단순 고정 지연: 언제나 33ms 고정 대기
        wait_ms = int(INTERVAL * 1000)
        
    # 키 입력 처리
    # [OpenCV 핵심] waitKey를 이용해 키 입력을 대기 및 입력된 키 값 추출
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = "Simple"
        print("[설정] Simple Mode (고정 딜레이)")
    elif key == ord('2'):
        mode = "Precision"
        print("[설정] Precision Mode (동적 보정 딜레이)")
    elif key == ord('b'):
        workload = not workload
        print(f"[설정] 가상 연산 부하: {'ON' if workload else 'OFF'}")

cv2.destroyAllWindows()
