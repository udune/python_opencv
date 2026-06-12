import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

def apply_size_mode(img, target_w, target_h, mode):
    # Ensure img is a numpy array
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Get image dimensions
    if len(img.shape) == 3:
        h, w, ch = img.shape
    else:
        h, w = img.shape
        ch = 1
        
    # Standardize size mode names
    mode_clean = mode.split(" ")[0].strip()
    
    if mode_clean == "Normal":
        # Create blank canvas matching channels
        if len(img.shape) == 3:
            canvas = np.zeros((target_h, target_w, ch), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        canvas[:copy_h, :copy_w] = img[:copy_h, :copy_w]
        return canvas
        
    elif mode_clean == "CenterImage":
        if len(img.shape) == 3:
            canvas = np.zeros((target_h, target_w, ch), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Calculate source crop regions if image is larger than canvas
        src_x = max((w - target_w) // 2, 0)
        src_y = max((h - target_h) // 2, 0)
        
        # Calculate destination start coordinates if image is smaller than canvas
        dest_x = max((target_w - w) // 2, 0)
        dest_y = max((target_h - h) // 2, 0)
        
        copy_w = min(w, target_w)
        copy_h = min(h, target_h)
        
        canvas[dest_y:dest_y+copy_h, dest_x:dest_x+copy_w] = img[src_y:src_y+copy_h, src_x:src_x+copy_w]
        return canvas
        
    elif mode_clean == "StretchImage":
        return cv2.resize(img, (target_w, target_h))
        
    elif mode_clean == "Zoom":
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            if len(img.shape) == 3:
                return np.zeros((target_h, target_w, ch), dtype=np.uint8)
            else:
                return np.zeros((target_h, target_w), dtype=np.uint8)
        
        resized = cv2.resize(img, (nw, nh))
        
        pad_w, pad_h = target_w - nw, target_h - nh
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
    else:
        return cv2.resize(img, (target_w, target_h))

def ndarray_to_qpixmap(img):
    if img is None:
        return QPixmap()
    
    if len(img.shape) == 3:
        h, w, ch = img.shape
        # OpenCV uses BGR, PyQt QImage wants RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        # Keep reference to rgb_img.data during QImage construction
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # fromImage copies data internally, so it's safe to dispose rgb_img after this
        return QPixmap.fromImage(qimg)
    else:
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
