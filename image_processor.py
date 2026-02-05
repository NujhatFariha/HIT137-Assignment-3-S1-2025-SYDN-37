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

