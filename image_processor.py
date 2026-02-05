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

class ImageProcessorApp:
  
    
    def __init__(self, root: tk.Tk):
      
        self.root = root
        self.root.title(" Image Processing Studio by Group 37")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 600)
        
    
        try:
            
            pass
        except:
            pass
        

        self.processor = ImageProcessor()
        self.history = ImageHistory(max_history=20)
        

        self.current_filepath: Optional[str] = None
        self.zoom_level: float = 1.0
        self.is_modified: bool = False
        self.auto_save_enabled: bool = False
        self.preview_mode: bool = False
        

        self.canvas: Optional[tk.Canvas] = None
        self.status_bar: Optional[tk.Label] = None
        self.info_panel: Optional[scrolledtext.ScrolledText] = None
        self.history_listbox: Optional[tk.Listbox] = None
        
        self.blur_slider: Optional[tk.Scale] = None
        self.brightness_slider: Optional[tk.Scale] = None
        self.contrast_slider: Optional[tk.Scale] = None
        self.saturation_slider: Optional[tk.Scale] = None
        self.resize_slider: Optional[tk.Scale] = None
        

        self.photo: Optional[ImageTk.PhotoImage] = None
        
        # setup GUI
        self._setup_styles()
        self._setup_menu()
        self._setup_gui()
        self._setup_status_bar()
        self._bind_shortcuts()
        
        # welcome message
        self._show_welcome_message()
    
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use('clam')  
        
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Section.TLabelframe.Label', font=('Arial', 10, 'bold'))
        style.configure('Action.TButton', padding=5)
    
    def _setup_menu(self) -> None:
  
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.open_image, 
                            accelerator="Ctrl+O")
        file_menu.add_command(label="Open Recent", state='disabled')
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_image, 
                            accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_as_image, 
                            accelerator="Ctrl+Shift+S")
        file_menu.add_command(label="Export...", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Image Properties", command=self.show_image_properties)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app, 
                            accelerator="Alt+F4")
        
        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, 
                            accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, 
                            accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset to Original", command=self.reset_image, 
                            accelerator="Ctrl+R")
        edit_menu.add_command(label="Clear History", command=self.clear_history)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in, 
                            accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, 
                            accelerator="Ctrl+-")
        view_menu.add_command(label="Fit to Window", command=self.fit_to_window, 
                            accelerator="Ctrl+0")
        view_menu.add_separator()
        view_menu.add_command(label="Show Image Info", command=self.toggle_info_panel)
        view_menu.add_command(label="Show History Panel", command=self.toggle_history_panel)
        
        # Filters Menu
        filters_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filters", menu=filters_menu)
        filters_menu.add_command(label="Grayscale", command=self.apply_grayscale)
        filters_menu.add_command(label="Blur...", command=self.show_blur_dialog)
        filters_menu.add_command(label="Edge Detection", command=self.apply_edge_detection)
        filters_menu.add_command(label="Sharpen", command=self.apply_sharpen)
        filters_menu.add_separator()
        filters_menu.add_command(label="Sepia Tone", command=self.apply_sepia)
        filters_menu.add_command(label="Negative", command=self.apply_negative)
        filters_menu.add_command(label="Emboss", command=self.apply_emboss)
        
        # Adjustments Menu
        adjust_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Adjustments", menu=adjust_menu)
        adjust_menu.add_command(label="Brightness/Contrast...", 
                              command=self.show_brightness_contrast_dialog)
        adjust_menu.add_command(label="Saturation...", command=self.show_saturation_dialog)
        
        # Transform Menu
        transform_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Transform", menu=transform_menu)
        transform_menu.add_command(label="Rotate 90° CW", 
                                 command=lambda: self.rotate_image(90))
        transform_menu.add_command(label="Rotate 180°", 
                                 command=lambda: self.rotate_image(180))
        transform_menu.add_command(label="Rotate 90° CCW", 
                                 command=lambda: self.rotate_image(270))
        transform_menu.add_separator()
        transform_menu.add_command(label="Flip Horizontal", 
                                 command=lambda: self.flip_image('horizontal'))
        transform_menu.add_command(label="Flip Vertical", 
                                 command=lambda: self.flip_image('vertical'))
        transform_menu.add_separator()
        transform_menu.add_command(label="Resize...", command=self.show_resize_dialog)
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
    
    def _setup_gui(self) -> None:

        main_container = ttk.Frame(self.root, padding="5")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        left_panel = self._create_control_panel(paned_window)
        paned_window.add(left_panel, weight=1)
        
        center_panel = self._create_image_panel(paned_window)
        paned_window.add(center_panel, weight=3)
        
        right_panel = self._create_info_panel(paned_window)
        paned_window.add(right_panel, weight=1)
    
    def _create_control_panel(self, parent) -> ttk.Frame:
      
        panel = ttk.Frame(parent, width=320)
        
        # Title
        title = ttk.Label(panel, text="Processing Controls", style='Title.TLabel')
        title.pack(pady=10)
        
        # Creating sllable frame
        canvas = tk.Canvas(panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
    
        canvas.pack(side="left", fill="both", expand=True, padx=(5, 0))
        scrollbar.pack(side="right", fill="y")
        

        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Filters", 
                                    style='Section.TLabelframe', padding=10)
        basic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(basic_frame, text="Grayscale", command=self.apply_grayscale,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(basic_frame, text="Edge Detection", command=self.apply_edge_detection,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(basic_frame, text="Sharpen", command=self.apply_sharpen,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        
   
        blur_frame = ttk.LabelFrame(scrollable_frame, text="Blur Effect", 
                                   style='Section.TLabelframe', padding=10)
        blur_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(blur_frame, text="Intensity:").pack()
        self.blur_slider = ttk.Scale(blur_frame, from_=1, to=15, orient=tk.HORIZONTAL)
        self.blur_slider.set(5)
        self.blur_slider.pack(fill=tk.X, pady=2)
        
        blur_value_label = ttk.Label(blur_frame, text="5")
        blur_value_label.pack()
        
        def update_blur_label(val):
            blur_value_label.config(text=f"{int(float(val))}")
        
        self.blur_slider.config(command=update_blur_label)
        
        ttk.Button(blur_frame, text="Apply Blur", command=self.apply_blur,
                  style='Action.TButton').pack(fill=tk.X, pady=5)
        
        # Brightness Section 
        brightness_frame = ttk.LabelFrame(scrollable_frame, text="Brightness", 
                                         style='Section.TLabelframe', padding=10)
        brightness_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(brightness_frame, text="Adjustment:").pack()
        self.brightness_slider = ttk.Scale(brightness_frame, from_=-100, to=100, 
                                          orient=tk.HORIZONTAL)
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=tk.X, pady=2)
        
        brightness_value_label = ttk.Label(brightness_frame, text="0")
        brightness_value_label.pack()
        
        def update_brightness_label(val):
            brightness_value_label.config(text=f"{int(float(val)):+d}")
        
        self.brightness_slider.config(command=update_brightness_label)
        
        ttk.Button(brightness_frame, text="Apply Brightness", 
                  command=self.apply_brightness, style='Action.TButton').pack(fill=tk.X, pady=5)
        
        # Contrast Section 
        contrast_frame = ttk.LabelFrame(scrollable_frame, text="Contrast", 
                                       style='Section.TLabelframe', padding=10)
        contrast_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(contrast_frame, text="Adjustment:").pack()
        self.contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=3.0, 
                                        orient=tk.HORIZONTAL)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X, pady=2)
        
        contrast_value_label = ttk.Label(contrast_frame, text="1.0x")
        contrast_value_label.pack()
        
        def update_contrast_label(val):
            contrast_value_label.config(text=f"{float(val):.1f}x")
        
        self.contrast_slider.config(command=update_contrast_label)
        
        ttk.Button(contrast_frame, text="Apply Contrast", command=self.apply_contrast,
                  style='Action.TButton').pack(fill=tk.X, pady=5)
        
        # Saturation Section 
        saturation_frame = ttk.LabelFrame(scrollable_frame, text="Saturation", 
                                         style='Section.TLabelframe', padding=10)
        saturation_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(saturation_frame, text="Adjustment:").pack()
        self.saturation_slider = ttk.Scale(saturation_frame, from_=0.0, to=2.0, 
                                          orient=tk.HORIZONTAL)
        self.saturation_slider.set(1.0)
        self.saturation_slider.pack(fill=tk.X, pady=2)
        
        saturation_value_label = ttk.Label(saturation_frame, text="1.0x")
        saturation_value_label.pack()
        
        def update_saturation_label(val):
            saturation_value_label.config(text=f"{float(val):.1f}x")
        
        self.saturation_slider.config(command=update_saturation_label)
        
        ttk.Button(saturation_frame, text="Apply Saturation", 
                  command=self.apply_saturation, style='Action.TButton').pack(fill=tk.X, pady=5)
        
        # Artistic Effects Section 
        artistic_frame = ttk.LabelFrame(scrollable_frame, text="Artistic Effects", 
                                       style='Section.TLabelframe', padding=10)
        artistic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(artistic_frame, text="Sepia Tone", command=self.apply_sepia,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(artistic_frame, text="Negative", command=self.apply_negative,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(artistic_frame, text="Emboss", command=self.apply_emboss,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        
        # Rotation Section 
        rotation_frame = ttk.LabelFrame(scrollable_frame, text="Rotation", 
                                       style='Section.TLabelframe', padding=10)
        rotation_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(rotation_frame, text="Rotate 90° CW", 
                  command=lambda: self.rotate_image(90),
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(rotation_frame, text="Rotate 180°", 
                  command=lambda: self.rotate_image(180),
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(rotation_frame, text="Rotate 90° CCW", 
                  command=lambda: self.rotate_image(270),
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        
        # Flip Section 
        flip_frame = ttk.LabelFrame(scrollable_frame, text="Flip", 
                                   style='Section.TLabelframe', padding=10)
        flip_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(flip_frame, text="Flip Horizontal", 
                  command=lambda: self.flip_image('horizontal'),
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(flip_frame, text="Flip Vertical", 
                  command=lambda: self.flip_image('vertical'),
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        
        # Resize Section 
        resize_frame = ttk.LabelFrame(scrollable_frame, text="Resize", 
                                     style='Section.TLabelframe', padding=10)
        resize_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(resize_frame, text="Scale (%):").pack()
        self.resize_slider = ttk.Scale(resize_frame, from_=10, to=200, 
                                      orient=tk.HORIZONTAL)
        self.resize_slider.set(100)
        self.resize_slider.pack(fill=tk.X, pady=2)
        
        resize_value_label = ttk.Label(resize_frame, text="100%")
        resize_value_label.pack()
        
        def update_resize_label(val):
            resize_value_label.config(text=f"{int(float(val))}%")
        
        self.resize_slider.config(command=update_resize_label)
        
        ttk.Button(resize_frame, text="Apply Resize", command=self.apply_resize,
                  style='Action.TButton').pack(fill=tk.X, pady=5)
        
        # Quick Actions 
        actions_frame = ttk.LabelFrame(scrollable_frame, text="Quick Actions", 
                                      style='Section.TLabelframe', padding=10)
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(actions_frame, text="Reset to Original", 
                  command=self.reset_image,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Save Image", 
                  command=self.save_image,
                  style='Action.TButton').pack(fill=tk.X, pady=2)
        
        return panel
    
    def _create_image_panel(self, parent) -> ttk.Frame:
   
        panel = ttk.Frame(parent)
        
        # Toolbar
        toolbar = ttk.Frame(panel)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Open", command=self.open_image, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_image, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="Undo", command=self.undo, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="Zoom In", command=self.zoom_in, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zoom Out", command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit", command=self.fit_to_window, width=10).pack(side=tk.LEFT, padx=2)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=0)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, 
                                   command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, 
                                   command=self.canvas.yview)
        
        self.canvas.config(xscrollcommand=h_scrollbar.set, 
                          yscrollcommand=v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        

        self.canvas.create_text(
            400, 300,
            text="Open an image to begin\n(File → Open or Ctrl+O)",
            font=("Arial", 16),
            fill="white",
            tags="placeholder"
        )
        
        return panel
    
    def _create_info_panel(self, parent) -> ttk.Frame:
        """Create the right info and history panel."""
        panel = ttk.Frame(parent, width=300)
        
 
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
     
        info_tab = ttk.Frame(notebook)
        notebook.add(info_tab, text="Image Info")
        
        self.info_panel = scrolledtext.ScrolledText(info_tab, wrap=tk.WORD, 
                                                    height=10, state='disabled')
        self.info_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History Tab
        history_tab = ttk.Frame(notebook)
        notebook.add(history_tab, text="History")
        
        history_label = ttk.Label(history_tab, text="Operation History:")
        history_label.pack(pady=5)
        
        self.history_listbox = tk.Listbox(history_tab, height=15)
        history_scrollbar = ttk.Scrollbar(history_tab, orient=tk.VERTICAL,
                                         command=self.history_listbox.yview)
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        

        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Processing Log")
        
        self.log_panel = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, 
                                                   height=10, state='disabled')
        self.log_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return panel



    #...................................................................
    
    def _setup_status_bar(self) -> None:
  
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        

        self.status_bar = ttk.Label(status_frame, text="Ready", anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
 
        self.zoom_label = ttk.Label(status_frame, text="100%", width=8)
        self.zoom_label.pack(side=tk.RIGHT, padx=5)
        
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        

        self.history_position_label = ttk.Label(status_frame, text="", width=10)
        self.history_position_label.pack(side=tk.RIGHT, padx=5)
        
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        
       
        self.memory_label = ttk.Label(status_frame, text="", width=15)
        self.memory_label.pack(side=tk.RIGHT, padx=5)
    
    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts for quick access."""
        shortcuts = {
            '<Control-o>': lambda e: self.open_image(),
            '<Control-s>': lambda e: self.save_image(),
            '<Control-Shift-S>': lambda e: self.save_as_image(),
            '<Control-z>': lambda e: self.undo(),
            '<Control-y>': lambda e: self.redo(),
            '<Control-r>': lambda e: self.reset_image(),
            '<Control-plus>': lambda e: self.zoom_in(),
            '<Control-minus>': lambda e: self.zoom_out(),
            '<Control-0>': lambda e: self.fit_to_window(),
            '<F1>': lambda e: self.show_user_guide(),
        }
        
        for key, command in shortcuts.items():
            self.root.bind(key, command)
    
    def _show_welcome_message(self) -> None:
       
        self._update_info_panel(
            "Welcome to image processing studio by group 37"
        )
    
    #Display Methods 
    
    def display_image(self) -> None:
        """Display the current image on canvas with zoom support."""
        image = self.processor.get_current_image()
        if image is None:
            return
        

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
     
        if self.zoom_level != 1.0:
            new_width = int(image_rgb.shape[1] * self.zoom_level)
            new_height = int(image_rgb.shape[0] * self.zoom_level)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
   
        pil_image = Image.fromarray(image_rgb)
        
  
        self.photo = ImageTk.PhotoImage(pil_image)
        
    
        self.canvas.delete("all")
        
      
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="image")
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
   
        self.update_status()
        self.update_image_info()
        self.update_history_display()
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def update_status(self, message: Optional[str] = None) -> None:
       
        if message:
            self.status_bar.config(text=message)
            self._log_message(message)
        else:
            info = self.processor.get_image_info()
            if info:
                status_text = (f"File: {info['filename']} | "
                             f"{info['width']}×{info['height']}px | "
                             f"{info['channels']} channels | "
                             f"{info['memory_size'] / 1024 / 1024:.2f} MB")
                self.status_bar.config(text=status_text)
            else:
                self.status_bar.config(text="Ready")
        
  
        if self.processor.get_current_image() is not None:
            current, total = self.history.get_current_position()
            self.history_position_label.config(text=f"{current}/{total}")
            
    
            memory_mb = self.history.get_memory_usage() / 1024 / 1024
            self.memory_label.config(text=f"History: {memory_mb:.1f} MB")
    
    def update_image_info(self) -> None:

        info = self.processor.get_image_info()
        if info:
            info_text = f"""Image Information
{'=' * 40}
Filename: {info['filename']}
Dimensions: {info['width']} × {info['height']} pixels
Channels: {info['channels']}
Total Pixels: {info['total_pixels']:,}
Memory Size: {info['memory_size'] / 1024 / 1024:.2f} MB
Data Type: {info['dtype']}

"""
            if info.get('original_dimensions'):
                orig_w, orig_h = info['original_dimensions']
                info_text += f"Original Size: {orig_w} × {orig_h} pixels\n"
            
            if info.get('file_size'):
                info_text += f"File Size: {info['file_size'] / 1024:.2f} KB\n"
            
            if info.get('load_time'):
                info_text += f"Loaded: {info['load_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            self._update_info_panel(info_text)
    
    def update_history_display(self) -> None:

        self.history_listbox.delete(0, tk.END)
        
        history_list = self.history.get_history_list()
        current, total = self.history.get_current_position()
        
        for i, operation in enumerate(history_list):
            marker = "→ " if i == current - 1 else "  "
            self.history_listbox.insert(tk.END, f"{marker}{i + 1}. {operation}")
            
            if i == current - 1:
                self.history_listbox.itemconfig(i, bg='lightblue')
    
    def _update_info_panel(self, text: str) -> None:
        """Update the info panel with new text."""
        self.info_panel.config(state='normal')
        self.info_panel.delete(1.0, tk.END)
        self.info_panel.insert(1.0, text)
        self.info_panel.config(state='disabled')
    
    def _log_message(self, message: str) -> None:
        """Add a message to the processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_panel.config(state='normal')
        self.log_panel.insert(tk.END, log_entry)
        self.log_panel.see(tk.END)
        self.log_panel.config(state='disabled')
    
    def save_to_history(self, operation_name: str = "Operation") -> None:
        """Save current image state to history."""
        image = self.processor.get_current_image()
        if image is not None:
            self.history.add_state(image, operation_name)
            self.is_modified = True
    
    
    def open_image(self) -> None:
    
        if self.is_modified:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before opening a new image?"
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_image()
        
        filetypes = [
            ("All Supported Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("JPEG Images", "*.jpg *.jpeg"),
            ("PNG Images", "*.png"),
            ("BMP Images", "*.bmp"),
            ("TIFF Images", "*.tiff *.tif"),
            ("GIF Images", "*.gif"),
            ("All Files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.processor.load_image(filepath)
                self.current_filepath = filepath
                self.history.clear()
                self.save_to_history("Image loaded")
                self.zoom_level = 1.0
                self.display_image()
                self.update_status(f"Loaded: {os.path.basename(filepath)}")
                self.is_modified = False
                self._log_message(f"Successfully loaded: {filepath}")
            except FileNotFoundError:
                messagebox.showerror("Error", "File not found")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self._log_message(f"Error loading image: {str(e)}")
    
    def save_image(self) -> None:
       
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        if self.current_filepath:
            try:
                self.processor.save_image(self.current_filepath, quality=95)
                self.update_status(f"Saved: {os.path.basename(self.current_filepath)}")
                messagebox.showinfo("Success", "Image saved successfully!")
                self.is_modified = False
                self._log_message(f"Saved to: {self.current_filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
                self._log_message(f"Error saving: {str(e)}")
        else:
            self.save_as_image()
    
    def save_as_image(self) -> None:
        """Save the image with a new filename."""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        filetypes = [
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("BMP Image", "*.bmp"),
            ("TIFF Image", "*.tiff"),
            ("All Files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.processor.save_image(filepath, quality=95)
                self.current_filepath = filepath
                self.update_status(f"Saved: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Image saved successfully!")
                self.is_modified = False
                self._log_message(f"Saved as: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
                self._log_message(f"Error saving: {str(e)}")
    
    def export_image(self) -> None:
       
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to export")
            return
        
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Options")
        export_dialog.geometry("400x250")
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        ttk.Label(export_dialog, text="Export Settings", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        

        format_frame = ttk.Frame(export_dialog)
        format_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        format_var = tk.StringVar(value="PNG")
        format_combo = ttk.Combobox(format_frame, textvariable=format_var,
                                   values=["PNG", "JPEG", "BMP", "TIFF"],
                                   state='readonly', width=15)
        format_combo.pack(side=tk.RIGHT)
        

        quality_frame = ttk.Frame(export_dialog)
        quality_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(quality_frame, text="Quality (JPEG):").pack(side=tk.LEFT)
        quality_var = tk.IntVar(value=95)
        quality_slider = ttk.Scale(quality_frame, from_=1, to=100, 
                                  variable=quality_var, orient=tk.HORIZONTAL)
        quality_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        quality_label = ttk.Label(export_dialog, text="95%")
        quality_label.pack()
        
        def update_quality_label(val):
            quality_label.config(text=f"{int(float(val))}%")
        
        quality_slider.config(command=update_quality_label)
        
        def do_export():
            format_ext = format_var.get().lower()
            if format_ext == "jpeg":
                format_ext = "jpg"
            
            filepath = filedialog.asksaveasfilename(
                title="Export Image",
                defaultextension=f".{format_ext}",
                filetypes=[(f"{format_var.get()} Image", f"*.{format_ext}")]
            )
            
            if filepath:
                try:
                    quality = quality_var.get() if format_ext == "jpg" else 95
                    self.processor.save_image(filepath, quality=quality)
                    messagebox.showinfo("Success", f"Image exported successfully to:\n{filepath}")
                    export_dialog.destroy()
                    self._log_message(f"Exported to: {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(export_dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Export", command=do_export).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=export_dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def exit_app(self) -> None:
        if self.is_modified:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?"
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_image()
        
        self.root.quit()
    
    # Edit Operations
    
    def undo(self) -> None:
   
        if self.history.can_undo():
            result = self.history.undo()
            if result:
                image, operation = result
                self.processor.current_image = image
                self.display_image()
                self.update_status(f"Undone: {operation}")
                self._log_message(f"Undo: {operation}")
        else:
            self.update_status("Nothing to undo")
    
    def redo(self) -> None:
       
        if self.history.can_redo():
            result = self.history.redo()
            if result:
                image, operation = result
                self.processor.current_image = image
                self.display_image()
                self.update_status(f"Redone: {operation}")
                self._log_message(f"Redo: {operation}")
        else:
            self.update_status("Nothing to redo")
    
    def reset_image(self) -> None:
    
        if self.processor.original_image is not None:
            if messagebox.askyesno("Reset Image", 
                                  "Reset to original image? This will clear all edits."):
                self.processor.reset_to_original()
                self.save_to_history("Reset to original")
                self.display_image()
                self.update_status("Reset to original image")
                self._log_message("Image reset to original")
    
    def clear_history(self) -> None:
       
        if messagebox.askyesno("Clear History", 
                              "Clear all history? This cannot be undone."):
            self.history.clear()
            if self.processor.get_current_image() is not None:
                self.save_to_history("History cleared")
            self.update_history_display()
            self.update_status("History cleared")
            self._log_message("History cleared")
            

