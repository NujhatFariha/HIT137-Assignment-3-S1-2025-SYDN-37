import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple


class ImageProcessor:
    
    def __init__(self):
       
        self._original_image: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._filename: Optional[str] = None
        self._filepath: Optional[str] = None
        self._processing_history: List[str] = []
        self._metadata: Dict[str, any] = {}
    
    @property
    def original_image(self) -> Optional[np.ndarray]:
        
        return self._original_image
    
    @property
    def current_image(self) -> Optional[np.ndarray]:
       
        return self._current_image
    
    @current_image.setter
    def current_image(self, value: np.ndarray):
     
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("Image must be a numpy array")
        self._current_image = value
    
    @property
    def filename(self) -> Optional[str]:
       
        return self._filename
    
    @property
    def processing_history(self) -> List[str]:
       
        return self._processing_history.copy()
    
    
    
    
    
    def load_image(self, filepath: str) -> bool:
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Unable to load image")
            
            self._original_image = image
            self._current_image = image.copy()
            self._filename = os.path.basename(filepath)
            self._filepath = filepath
            self._processing_history = [f"Loaded: {self._filename}"]
            
            # Storing metadata
            self._metadata = {
                'load_time': datetime.now(),
                'file_size': os.path.getsize(filepath),
                'original_dimensions': (image.shape[1], image.shape[0])
            }
            
            return True
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def save_image(self, filepath: str, quality: int = 95) -> bool:
        
        if self._current_image is None:
            return False
        
        try:
            # Determining file extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                params = []
            
            success = cv2.imwrite(filepath, self._current_image, params)
            
            if success:
                self._add_to_history(f"Saved: {os.path.basename(filepath)}")
            
            return success
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")
    
    def reset_to_original(self) -> None:

        if self._original_image is not None:
            self._current_image = self._original_image.copy()
            self._add_to_history("Reset to original")
    
    def convert_to_grayscale(self) -> None:
       
        if self._current_image is None:
            return
        
        gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
        self._current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._add_to_history("Applied grayscale conversion")
    
    def apply_blur(self, intensity: int = 5) -> None:
        
        if self._current_image is None:
            return
        
   
        intensity = max(1, min(15, intensity))
        kernel_size = intensity * 2 + 1
        
        self._current_image = cv2.GaussianBlur(
            self._current_image, 
            (kernel_size, kernel_size), 
            0
        )
        self._add_to_history(f"Applied Gaussian blur (intensity: {intensity})")
    
    def detect_edges(self, threshold1: int = 100, threshold2: int = 200) -> None:
       
        if self._current_image is None:
            return
        
        gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        self._current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self._add_to_history(f"Applied Canny edge detection")
    
    def adjust_brightness(self, value: int) -> None:
       
        if self._current_image is None:
            return
        
        value = max(-100, min(100, value))
        hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjusting V channel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self._add_to_history(f"Adjusted brightness ({value:+d})")
    
    def adjust_contrast(self, value: float) -> None:
        
        if self._current_image is None:
            return
        
        value = max(0.5, min(3.0, value))
        self._current_image = cv2.convertScaleAbs(self._current_image, alpha=value, beta=0)
        self._add_to_history(f"Adjusted contrast ({value:.1f}x)")
    
    def rotate_image(self, angle: int) -> None:
        
        if self._current_image is None:
            return
        
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        
        if angle in rotation_map:
            self._current_image = cv2.rotate(self._current_image, rotation_map[angle])
            self._add_to_history(f"Rotated {angle}°")
    
    def flip_image(self, direction: str) -> None:
       
        if self._current_image is None:
            return
        
        flip_code = 1 if direction == 'horizontal' else 0
        self._current_image = cv2.flip(self._current_image, flip_code)
        self._add_to_history(f"Flipped {direction}")
    
    def resize_image(self, scale_percent: int) -> None:
      
        if self._current_image is None:
            return
        
        scale_percent = max(10, min(200, scale_percent))
        width = int(self._current_image.shape[1] * scale_percent / 100)
        height = int(self._current_image.shape[0] * scale_percent / 100)
        
        self._current_image = cv2.resize(
            self._current_image, 
            (width, height), 
            interpolation=cv2.INTER_AREA if scale_percent < 100 else cv2.INTER_CUBIC
        )
        self._add_to_history(f"Resized to {scale_percent}%")
    
    def apply_sharpen(self) -> None:
    
        if self._current_image is None:
            return
        
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        self._current_image = cv2.filter2D(self._current_image, -1, kernel)
        self._add_to_history("Applied sharpening filter")
    
    def apply_emboss(self) -> None:
       
        if self._current_image is None:
            return
        
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        self._current_image = cv2.filter2D(self._current_image, -1, kernel)
        self._add_to_history("Applied emboss effect")
    
    def apply_sepia(self) -> None:

        if self._current_image is None:
            return
        
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        self._current_image = cv2.transform(self._current_image, kernel)
        self._current_image = np.clip(self._current_image, 0, 255).astype(np.uint8)
        self._add_to_history("Applied sepia tone")
    
    def apply_negative(self) -> None:

        if self._current_image is None:
            return
        
        self._current_image = cv2.bitwise_not(self._current_image)
        self._add_to_history("Applied negative effect")
    
    def adjust_saturation(self, value: float) -> None:
        
        if self._current_image is None:
            return
        
        value = max(0.0, min(2.0, value))
        hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
        hsv = hsv.astype(np.uint8)
        self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self._add_to_history(f"Adjusted saturation ({value:.1f}x)")
    
    def get_image_info(self) -> Optional[Dict[str, any]]:
        
        if self._current_image is None:
            return None
        
        height, width = self._current_image.shape[:2]
        channels = self._current_image.shape[2] if len(self._current_image.shape) > 2 else 1
        
        info = {
            'filename': self._filename,
            'filepath': self._filepath,
            'width': width,
            'height': height,
            'channels': channels,
            'total_pixels': width * height,
            'memory_size': self._current_image.nbytes,
            'dtype': str(self._current_image.dtype),
        }
        
        # Add original metadata if available
        if self._metadata:
            info.update({
                'file_size': self._metadata.get('file_size'),
                'load_time': self._metadata.get('load_time'),
                'original_dimensions': self._metadata.get('original_dimensions')
            })
        
        return info
    
    def get_current_image(self) -> Optional[np.ndarray]:
        
        return self._current_image
    
    def _add_to_history(self, operation: str) -> None:
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._processing_history.append(f"[{timestamp}] {operation}")










class ImageHistory:
   
    def __init__(self, max_history: int = 20):
       
        self._history: List[np.ndarray] = []
        self._current_index: int = -1
        self._max_history: int = max(5, min(50, max_history))  # Limit between 5-50
        self._operation_names: List[str] = []
    
    def add_state(self, image: np.ndarray, operation_name: str = "Operation") -> None:
       
        if image is None:
            return
        
        self._history = self._history[:self._current_index + 1]
        self._operation_names = self._operation_names[:self._current_index + 1]
        
        self._history.append(image.copy())
        self._operation_names.append(operation_name)
        
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._operation_names.pop(0)
        else:
            self._current_index += 1
    
    def can_undo(self) -> bool:
        
        return self._current_index > 0
    
    def can_redo(self) -> bool:

        return self._current_index < len(self._history) - 1
    
    def undo(self) -> Optional[Tuple[np.ndarray, str]]:
       
        if self.can_undo():
            self._current_index -= 1
            return (
                self._history[self._current_index].copy(),
                self._operation_names[self._current_index]
            )
        return None
    
    def redo(self) -> Optional[Tuple[np.ndarray, str]]:
       
        if self.can_redo():
            self._current_index += 1
            return (
                self._history[self._current_index].copy(),
                self._operation_names[self._current_index]
            )
        return None
    
    def get_history_list(self) -> List[str]:
        
        return self._operation_names.copy()
    
    def get_current_position(self) -> Tuple[int, int]:
       
        return (self._current_index + 1, len(self._history))
    
    def clear(self) -> None:
       
        self._history.clear()
        self._operation_names.clear()
        self._current_index = -1
    
    def get_memory_usage(self) -> int:
       
        total_bytes = sum(img.nbytes for img in self._history)
        return total_bytes
