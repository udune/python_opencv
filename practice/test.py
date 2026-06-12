import cv2

# 이미지 읽기 - 반환값은 numpy 배열(ndarray),
# 채널 순서는 BGR
image = cv2.imread("test.jpg")

if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

cv2.imshow("Window", image) # OS 창에 직접 출력
cv2.waitKey(0) # 키 입력 대기 (0 = 무한)
cv2.destroyAllWindows() # 모든 창 닫기
