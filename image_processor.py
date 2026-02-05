"""
Image Processing Desktop Application
Demonstrates OOP, Tkinter GUI, and OpenCV image processing
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class ImageProcessor:
    """
    Handles all image processing operations using OpenCV.
    Demonstrates: Encapsulation, Constructor, Methods
    """
    
    def __init__(self):
        """Initialize the ImageProcessor with default values"""
        self.original_image = None
        self.current_image = None
        self.filename = None
    
    def load_image(self, filepath):
        """Load an image from file"""
        try:
            self.original_image = cv2.imread(filepath)
            if self.original_image is None:
                raise ValueError("Unable to load image")
            self.current_image = self.original_image.copy()
            self.filename = os.path.basename(filepath)
            return True
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def save_image(self, filepath):
        """Save the current image to file"""
        if self.current_image is not None:
            cv2.imwrite(filepath, self.current_image)
            return True
        return False
    
    def reset_to_original(self):
        """Reset current image to original"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
    
    def convert_to_grayscale(self):
        """Convert image to grayscale"""
        if self.current_image is not None:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
    
    def apply_blur(self, intensity=5):
        """Apply Gaussian blur with adjustable intensity"""
        if self.current_image is not None:
            # Ensure kernel size is odd
            kernel_size = intensity * 2 + 1
            self.current_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
    
    def detect_edges(self, threshold1=100, threshold2=200):
        """Apply Canny edge detection"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def adjust_brightness(self, value):
        """Adjust image brightness (-100 to 100)"""
        if self.current_image is not None:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            if value >= 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = abs(value)
                v[v < lim] = 0
                v[v >= lim] -= abs(value)
            
            final_hsv = cv2.merge((h, s, v))
            self.current_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, value):
        """Adjust image contrast (0.5 to 3.0)"""
        if self.current_image is not None:
            self.current_image = cv2.convertScaleAbs(self.current_image, alpha=value, beta=0)
    
    def rotate_image(self, angle):
        """Rotate image by specified angle (90, 180, 270)"""
        if self.current_image is not None:
            if angle == 90:
                self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_180)
            elif angle == 270:
                self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def flip_image(self, direction):
        """Flip image horizontally or vertically"""
        if self.current_image is not None:
            if direction == 'horizontal':
                self.current_image = cv2.flip(self.current_image, 1)
            elif direction == 'vertical':
                self.current_image = cv2.flip(self.current_image, 0)
    
    def resize_image(self, scale_percent):
        """Resize image by percentage (10 to 200)"""
        if self.current_image is not None:
            width = int(self.current_image.shape[1] * scale_percent / 100)
            height = int(self.current_image.shape[0] * scale_percent / 100)
            self.current_image = cv2.resize(self.current_image, (width, height), interpolation=cv2.INTER_AREA)
    
    def get_image_info(self):
        """Get current image information"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            return {
                'filename': self.filename,
                'width': width,
                'height': height,
                'channels': self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            }
        return None
    
    def get_current_image(self):
        """Get the current image for display"""
        return self.current_image


class ImageHistory:
    """
    Manages undo/redo functionality for image processing.
    Demonstrates: Encapsulation, Constructor, Methods
    """
    
    def __init__(self, max_history=10):
        """Initialize history with maximum number of states to store"""
        self.history = []
        self.current_index = -1
        self.max_history = max_history
    
    def add_state(self, image):
        """Add a new image state to history"""
        if image is None:
            return
        
        # Remove any states after current index (when adding after undo)
        self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(image.copy())
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1
    
    def can_undo(self):
        """Check if undo is possible"""
        return self.current_index > 0
    
    def can_redo(self):
        """Check if redo is possible"""
        return self.current_index < len(self.history) - 1
    
    def undo(self):
        """Go back to previous state"""
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index].copy()
        return None
    
    def redo(self):
        """Go forward to next state"""
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index].copy()
        return None
    
    def clear(self):
        """Clear all history"""
        self.history = []
        self.current_index = -1


class ImageProcessorApp:
    """
    Main application class managing the GUI and coordinating between components.
    Demonstrates: Class Interaction, Encapsulation, Constructor, Methods
    """
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1200x800")
        
        # Initialize processor and history (Class Interaction)
        self.processor = ImageProcessor()
        self.history = ImageHistory()
        
        # Current file path for save operations
        self.current_filepath = None
        
        # Setup GUI components
        self.setup_menu()
        self.setup_gui()
        self.setup_status_bar()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-o>', lambda e: self.open_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
    
    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset to Original", command=self.reset_image)
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        self.setup_control_panel(main_container)
        
        # Right panel - Image display
        self.setup_image_display(main_container)
    
    def setup_control_panel(self, parent):
        """Create the control panel with all buttons and sliders"""
        control_frame = tk.Frame(parent, width=300, relief=tk.RAISED, borderwidth=1)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(control_frame, text="Image Processing Controls", 
                        font=("Arial", 12, "bold"))
        title.pack(pady=10)
        
        # Basic Filters Section
        self.create_section(control_frame, "Basic Filters", [
            ("Grayscale", self.apply_grayscale),
            ("Edge Detection", self.apply_edge_detection)
        ])
        
        # Blur Section with slider
        blur_frame = tk.LabelFrame(control_frame, text="Blur Effect", padx=10, pady=10)
        blur_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(blur_frame, text="Intensity:").pack()
        self.blur_slider = tk.Scale(blur_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                                    command=self.update_blur_preview)
        self.blur_slider.set(5)
        self.blur_slider.pack(fill=tk.X)
        tk.Button(blur_frame, text="Apply Blur", command=self.apply_blur).pack(pady=5)
        
        # Brightness Section with slider
        brightness_frame = tk.LabelFrame(control_frame, text="Brightness", padx=10, pady=10)
        brightness_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(brightness_frame, text="Adjustment:").pack()
        self.brightness_slider = tk.Scale(brightness_frame, from_=-100, to=100, 
                                         orient=tk.HORIZONTAL)
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=tk.X)
        tk.Button(brightness_frame, text="Apply Brightness", 
                 command=self.apply_brightness).pack(pady=5)
        
        # Contrast Section with slider
        contrast_frame = tk.LabelFrame(control_frame, text="Contrast", padx=10, pady=10)
        contrast_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(contrast_frame, text="Adjustment:").pack()
        self.contrast_slider = tk.Scale(contrast_frame, from_=0.5, to=3.0, 
                                       resolution=0.1, orient=tk.HORIZONTAL)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X)
        tk.Button(contrast_frame, text="Apply Contrast", 
                 command=self.apply_contrast).pack(pady=5)
        
        # Rotation Section
        self.create_section(control_frame, "Rotation", [
            ("Rotate 90°", lambda: self.rotate_image(90)),
            ("Rotate 180°", lambda: self.rotate_image(180)),
            ("Rotate 270°", lambda: self.rotate_image(270))
        ])
        
        # Flip Section
        self.create_section(control_frame, "Flip", [
            ("Flip Horizontal", lambda: self.flip_image('horizontal')),
            ("Flip Vertical", lambda: self.flip_image('vertical'))
        ])
        
        # Resize Section with slider
        resize_frame = tk.LabelFrame(control_frame, text="Resize", padx=10, pady=10)
        resize_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(resize_frame, text="Scale (%):").pack()
        self.resize_slider = tk.Scale(resize_frame, from_=10, to=200, 
                                     orient=tk.HORIZONTAL)
        self.resize_slider.set(100)
        self.resize_slider.pack(fill=tk.X)
        tk.Button(resize_frame, text="Apply Resize", 
                 command=self.apply_resize).pack(pady=5)
    
    def create_section(self, parent, title, buttons):
        """Helper method to create a section with multiple buttons"""
        frame = tk.LabelFrame(parent, text=title, padx=10, pady=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        for text, command in buttons:
            tk.Button(frame, text=text, command=command, width=20).pack(pady=2)
    
    def setup_image_display(self, parent):
        """Setup the image display area"""
        display_frame = tk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(display_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = tk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = tk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Initial message
        self.canvas.create_text(400, 300, text="Open an image to begin", 
                               font=("Arial", 16), fill="white", tags="placeholder")
    
    def setup_status_bar(self):
        """Create status bar at the bottom"""
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message=None):
        """Update status bar with image info or custom message"""
        if message:
            self.status_bar.config(text=message)
        else:
            info = self.processor.get_image_info()
            if info:
                status_text = f"File: {info['filename']} | Dimensions: {info['width']}x{info['height']} | Channels: {info['channels']}"
                self.status_bar.config(text=status_text)
            else:
                self.status_bar.config(text="Ready")
    
    def display_image(self):
        """Display the current image on canvas"""
        image = self.processor.get_current_image()
        if image is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Display image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            # Update status
            self.update_status()
    
    def save_to_history(self):
        """Save current state to history"""
        image = self.processor.get_current_image()
        if image is not None:
            self.history.add_state(image)
    
    # File operations
    def open_image(self):
        """Open an image file"""
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                self.processor.load_image(filepath)
                self.current_filepath = filepath
                self.history.clear()
                self.save_to_history()
                self.display_image()
                self.update_status(f"Loaded: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def save_image(self):
        """Save the current image"""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        if self.current_filepath:
            try:
                self.processor.save_image(self.current_filepath)
                self.update_status(f"Saved: {os.path.basename(self.current_filepath)}")
                messagebox.showinfo("Success", "Image saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        else:
            self.save_as_image()
    
    def save_as_image(self):
        """Save the image with a new filename"""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                self.processor.save_image(filepath)
                self.current_filepath = filepath
                self.update_status(f"Saved: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Image saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def exit_app(self):
        """Exit the application"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
    
    # Edit operations
    def undo(self):
        """Undo last operation"""
        if self.history.can_undo():
            image = self.history.undo()
            if image is not None:
                self.processor.current_image = image
                self.display_image()
                self.update_status("Undo successful")
        else:
            self.update_status("Nothing to undo")
    
    def redo(self):
        """Redo last undone operation"""
        if self.history.can_redo():
            image = self.history.redo()
            if image is not None:
                self.processor.current_image = image
                self.display_image()
                self.update_status("Redo successful")
        else:
            self.update_status("Nothing to redo")
    
    def reset_image(self):
        """Reset image to original"""
        if self.processor.original_image is not None:
            if messagebox.askyesno("Reset", "Reset to original image?"):
                self.processor.reset_to_original()
                self.save_to_history()
                self.display_image()
                self.update_status("Reset to original")
    
    # Image processing operations
    def apply_grayscale(self):
        """Apply grayscale filter"""
        if self.check_image_loaded():
            self.save_to_history()
            self.processor.convert_to_grayscale()
            self.display_image()
            self.update_status("Applied grayscale filter")
    
    def apply_blur(self):
        """Apply blur effect"""
        if self.check_image_loaded():
            self.save_to_history()
            intensity = self.blur_slider.get()
            self.processor.apply_blur(intensity)
            self.display_image()
            self.update_status(f"Applied blur (intensity: {intensity})")
    
    def update_blur_preview(self, value):
        """Update blur slider label"""
        pass  # Could add real-time preview here
    
    def apply_edge_detection(self):
        """Apply edge detection"""
        if self.check_image_loaded():
            self.save_to_history()
            self.processor.detect_edges()
            self.display_image()
            self.update_status("Applied edge detection")
    
    def apply_brightness(self):
        """Apply brightness adjustment"""
        if self.check_image_loaded():
            self.save_to_history()
            value = self.brightness_slider.get()
            self.processor.adjust_brightness(value)
            self.display_image()
            self.update_status(f"Adjusted brightness ({value:+d})")
    
    def apply_contrast(self):
        """Apply contrast adjustment"""
        if self.check_image_loaded():
            self.save_to_history()
            value = self.contrast_slider.get()
            self.processor.adjust_contrast(value)
            self.display_image()
            self.update_status(f"Adjusted contrast ({value:.1f})")
    
    def rotate_image(self, angle):
        """Rotate image by specified angle"""
        if self.check_image_loaded():
            self.save_to_history()
            self.processor.rotate_image(angle)
            self.display_image()
            self.update_status(f"Rotated {angle}°")
    
    def flip_image(self, direction):
        """Flip image in specified direction"""
        if self.check_image_loaded():
            self.save_to_history()
            self.processor.flip_image(direction)
            self.display_image()
            self.update_status(f"Flipped {direction}")
    
    def apply_resize(self):
        """Apply resize operation"""
        if self.check_image_loaded():
            self.save_to_history()
            scale = self.resize_slider.get()
            self.processor.resize_image(scale)
            self.display_image()
            self.update_status(f"Resized to {scale}%")
    
    def check_image_loaded(self):
        """Check if an image is loaded"""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return False
        return True


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
