import os
import cv2
import numpy as np

class OpenCVClass:
    """
    OpenCVClass wraps the raw BGR and Grayscale images as NumPy ndarrays.
    It provides methods to load, convert, and dispose resources,
    simulating native memory lifecycle management in Python.
    """
    def __init__(self):
        self._src_image = None
        self._gray_image = None

    @property
    def src_image(self):
        return self._src_image

    @property
    def gray_image(self):
        return self._gray_image

    def load_image(self, file_path):
        # Clean up existing source image first
        self.release_src_image()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"지정된 파일을 찾을 수 없습니다: {file_path}")

        # Read image supporting Unicode paths
        try:
            img_array = np.fromfile(file_path, np.uint8)
            self._src_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise IOError(f"이미지 로드 중 오류가 발생했습니다: {str(e)}")

        if self._src_image is None:
            raise IOError("이미지 디코딩에 실패했습니다.")

    def convert_to_gray(self):
        if self._src_image is None:
            raise ValueError("원본 이미지가 로드되지 않았습니다.")

        # Clean up existing gray image first
        self.release_gray_image()

        try:
            self._gray_image = cv2.cvtColor(self._src_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise RuntimeError(f"Grayscale 변환 중 오류가 발생했습니다: {str(e)}")

    def release_src_image(self):
        # Drop reference to free numpy array memory
        self._src_image = None

    def release_gray_image(self):
        # Drop reference to free numpy array memory
        self._gray_image = None

    def dispose(self):
        # Clean up all image resources
        self.release_src_image()
        self.release_gray_image()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()
