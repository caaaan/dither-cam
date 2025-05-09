import os
import sys
import time
import queue
import threading
import traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QComboBox, QFileDialog,
                             QRadioButton, QSpinBox, QGroupBox, QTabWidget, 
                             QStackedWidget, QCheckBox, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QColor

# Import from classes folder
from classes.image_viewer import ImageViewer
from classes.helper import (fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither, 
                            block_average_rgb, block_average_gray, nearest_upscale_rgb, 
                            nearest_upscale_gray, downscale_dither_upscale)

class DitherApp(QMainWindow):
    """Main application window for the dithering app"""
    
    def __init__(self):
        super().__init__()
        
        # State variables
        self.original_image = None  # Original PIL image loaded from file
        self.dithered_image = None  # Currently displayed dithered PIL image
        self.last_dither_time = 0   # Track when we last dithered to avoid excessive processing
        self.is_camera_active = False  # Flag to track if camera mode is active
        self.camera_thread = None   # Thread for camera capture
        self.processing_thread = None  # Thread for processing frames
        self.frame_queue = queue.Queue(maxsize=3)  # Small queue to avoid memory issues
        
        # Camera tab flag
        self.has_camera_tab = True
        
        # Setup UI
        self.init_ui()
        
        # Create a timer for periodic UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # 100ms = 10 updates/second
        
        # Initialize "current" settings to detect changes
        self._current_pixel_scale = self.scale_slider.value()
        self._last_camera_error_time = 0
        
        print("Application initialized")
        
    def init_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Dither Cam")
        self.resize(1000, 700)
        
        # Set up tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.file_tab = QWidget()
        self.camera_tab = QWidget()
        
        # Add tabs
        self.tabs.addTab(self.file_tab, "File")
        if self.has_camera_tab:
            self.tabs.addTab(self.camera_tab, "Camera")
        
        # Tab change event
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Setup file tab
        self.setup_file_tab()
        
        # Setup camera tab if available
        if self.has_camera_tab:
            self.setup_camera_tab()
        
        # Set up shared controls
        self.setup_controls()
        
        # Show UI
        self.show()
    
    def setup_file_tab(self):
        """Initialize the file tab UI"""
        layout = QVBoxLayout()
        
        # Top buttons for file operations
        button_layout = QHBoxLayout()
        
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_button)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.zoom_changed.connect(self.on_zoom_changed)
        
        # Main content area with image viewer
        layout.addWidget(self.image_viewer)
        
        # Set layout
        self.file_tab.setLayout(layout)
    
    def setup_camera_tab(self):
        """Initialize the camera tab UI"""
        layout = QVBoxLayout()
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.camera_button)
        
        self.screenshot_button = QPushButton("Take Screenshot")
        self.screenshot_button.clicked.connect(self.take_screenshot)
        self.screenshot_button.setEnabled(False)  # Initially disabled
        camera_controls.addWidget(self.screenshot_button)
        
        layout.addLayout(camera_controls)
        
        # Camera viewer
        self.camera_viewer = ImageViewer()
        self.camera_viewer.zoom_changed.connect(self.on_zoom_changed)
        
        # Main content area with camera viewer
        layout.addWidget(self.camera_viewer)
        
        # Set layout
        self.camera_tab.setLayout(layout)
    
    def setup_controls(self):
        """Set up the control sidebar"""
        # Create a dock widget for controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(250)  # Set fixed width for controls
        
        # Add controls to the sidebar
        controls_layout.addWidget(QLabel("<b>Dithering Controls</b>"))
        
        # Algorithm selector
        controls_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Floyd-Steinberg", "Simple Threshold"])
        self.algorithm_combo.currentIndexChanged.connect(self.on_settings_changed)
        controls_layout.addWidget(self.algorithm_combo)
        
        # Color mode
        color_group = QGroupBox("Color Mode")
        color_layout = QVBoxLayout()
        
        self.rgb_mode = QRadioButton("RGB Color")
        self.rgb_mode.setChecked(True)
        self.rgb_mode.toggled.connect(self.on_settings_changed)
        color_layout.addWidget(self.rgb_mode)
        
        self.bw_mode = QRadioButton("Black & White")
        self.bw_mode.toggled.connect(self.on_settings_changed)
        color_layout.addWidget(self.bw_mode)
        
        color_group.setLayout(color_layout)
        controls_layout.addWidget(color_group)
        
        # Threshold slider
        controls_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.on_settings_changed)
        controls_layout.addWidget(self.threshold_slider)
        self.threshold_value = QLabel("128")
        controls_layout.addWidget(self.threshold_value)
        
        # Contrast slider
        controls_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_settings_changed)
        controls_layout.addWidget(self.contrast_slider)
        self.contrast_value = QLabel("100%")
        controls_layout.addWidget(self.contrast_value)
        
        # Pixel scale slider
        controls_layout.addWidget(QLabel("Pixel Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(8)
        self.scale_slider.setValue(1)
        self.scale_slider.valueChanged.connect(self.on_settings_changed)
        controls_layout.addWidget(self.scale_slider)
        self.scale_value = QLabel("1x")
        controls_layout.addWidget(self.scale_value)
        
        # Zoom controls
        controls_layout.addWidget(QLabel("Zoom:"))
        zoom_layout = QHBoxLayout()
        
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_button)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10% zoom
        self.zoom_slider.setMaximum(500)  # 500% zoom
        self.zoom_slider.setValue(100)  # 100% = original size
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_button)
        
        controls_layout.addLayout(zoom_layout)
        self.zoom_value = QLabel("100%")
        controls_layout.addWidget(self.zoom_value)
        
        # Performance monitor
        self.perf_label = QLabel("Processing time: -")
        controls_layout.addWidget(self.perf_label)
        
        # Add spacer
        controls_layout.addStretch()
        
        # Credits
        credits = QLabel("Dither Cam v1.0")
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(credits)
        
        # Add the control panel to the main window
        main_layout = QHBoxLayout()
        
        # Use a stacked widget to hold the tabs
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.tabs)
        
        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(controls_widget)
        
        # Create a central widget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def on_settings_changed(self):
        """Handle any change in dithering settings"""
        # Update UI labels
        self.threshold_value.setText(str(self.threshold_slider.value()))
        self.contrast_value.setText(f"{self.contrast_slider.value()}%")
        self.scale_value.setText(f"{self.scale_slider.value()}x")
        
        # If camera mode is active, check if we need to adjust camera resolution
        if self.is_camera_active and self.camera_thread is not None:
            current_scale = self.scale_slider.value()
            if current_scale != self._current_pixel_scale:
                print(f"Pixel scale changed from {self._current_pixel_scale} to {current_scale}")
                self._current_pixel_scale = current_scale
                
                # Try to reconfigure camera resolution if supported
                if hasattr(self.camera_thread, 'reconfigure_resolution'):
                    success = self.camera_thread.reconfigure_resolution(current_scale)
                    if not success:
                        print("Camera resolution adjustment failed")
        
        # Redither the current image if we're in file mode
        self.update_dithering()
    
    def on_zoom_changed(self, new_zoom):
        """Handle zoom changes from the image viewer"""
        # Update the UI to reflect the new zoom level
        zoom_percent = int(new_zoom * 100)
        self.zoom_value.setText(f"{zoom_percent}%")
        
        # Update the slider (without triggering callbacks)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_percent)
        self.zoom_slider.blockSignals(False)
    
    def on_zoom_slider_changed(self):
        """Handle changes to the zoom slider"""
        zoom_level = self.zoom_slider.value() / 100.0
        self.zoom_value.setText(f"{self.zoom_slider.value()}%")
        
        # Apply zoom to the currently visible viewer
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0:
            self.image_viewer.set_zoom_level(zoom_level)
        elif current_tab_index == 1 and self.has_camera_tab:
            self.camera_viewer.set_zoom_level(zoom_level)
    
    def zoom_in(self):
        """Increase zoom level"""
        current_value = self.zoom_slider.value()
        new_value = min(current_value * 1.1, self.zoom_slider.maximum())
        self.zoom_slider.setValue(int(new_value))
    
    def zoom_out(self):
        """Decrease zoom level"""
        current_value = self.zoom_slider.value()
        new_value = max(current_value * 0.9, self.zoom_slider.minimum())
        self.zoom_slider.setValue(int(new_value))
    
    def on_tab_changed(self, index):
        """Handle switching between file and camera tabs"""
        print(f"Tab changed to {index}")
        
        # Stop camera if we're leaving the camera tab
        if index == 0 and self.is_camera_active:
            print("Stopping camera because we switched to file tab")
            self.stop_camera()
    
    def open_image(self):
        """Open an image file and display it"""
        # If camera is running, stop it first
        if self.is_camera_active:
            self.stop_camera()
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Load image
                self.original_image = Image.open(file_path)
                print(f"Loaded image: {self.original_image.format}, {self.original_image.size}, {self.original_image.mode}")
                
                # Apply dithering
                self.update_dithering()
                
                # Switch to file tab if we're not already there
                self.tabs.setCurrentIndex(0)
                
            except Exception as e:
                print(f"Error loading image: {e}")
                traceback.print_exc()
    
    def save_image(self):
        """Save the current dithered image to a file"""
        # Check if we have a dithered image to save (either from file or camera)
        if not self.dithered_image:
            print("No image to save")
            return
        
        try:
            # Debug info about the image
            print(f"Image to save: type={type(self.dithered_image)}, "
                  f"mode={self.dithered_image.mode if hasattr(self.dithered_image, 'mode') else 'unknown'}")
            
            # Get save path with explicit file extension filters
            path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
            )
            
            # Handle case where user cancels
            if not path:
                return
                
            # Make sure image is in RGB mode for wider compatibility
            image_to_save = self.dithered_image
            if hasattr(image_to_save, 'mode') and image_to_save.mode == 'RGBA':
                # Convert RGBA to RGB for better compatibility
                rgb_image = Image.new('RGB', image_to_save.size, (255, 255, 255))
                rgb_image.paste(image_to_save, mask=image_to_save.split()[3])
                image_to_save = rgb_image
            
            # Add extension if missing
            if '.' not in path:
                if 'png' in selected_filter.lower():
                    path += '.png'
                elif 'jpg' in selected_filter.lower() or 'jpeg' in selected_filter.lower():
                    path += '.jpg'
            
            # Save image
            image_to_save.save(path)
            print(f"Image saved to {path}")
            
        except Exception as e:
            print(f"Error saving image: {e}")
            traceback.print_exc()
    
    def update_dithering(self):
        """Apply dithering to the current image based on current settings"""
        # If we're in camera mode, don't process here (the frame processor does it)
        if self.is_camera_active:
            return
            
        # Check if we have an image to process
        if not self.original_image:
            return
            
        # Throttle processing to avoid excessive updates
        current_time = time.time()
        if current_time - self.last_dither_time < 0.1:  # Max 10 updates per second
            return
            
        self.last_dither_time = current_time
        
        # Apply dithering with current settings
        try:
            process_start = time.time()
            
            # Get settings
            alg = self.algorithm_combo.currentText()
            thr = self.threshold_slider.value()
            pixel_s = self.scale_slider.value()
            
            # Apply selected algorithm
            if alg == "Floyd-Steinberg":
                self.dithered_image = self.floyd_steinberg_numpy(self.original_image, thr, pixel_s)
            elif alg == "Simple Threshold":
                self.dithered_image = self.simple_threshold(self.original_image, thr, pixel_s)
            
            # Calculate and display performance info
            process_time = (time.time() - process_start) * 1000  # ms
            self.perf_label.setText(f"Processing time: {process_time:.1f}ms")
            
            # Display result
            self.display_image(self.dithered_image)
            
        except Exception as e:
            print(f"Error applying dithering: {e}")
            traceback.print_exc()
    
    def display_image(self, pil_image):
        if not pil_image:
            self.image_viewer.set_image(None) # set_image(None) handles zoom reset
            return
        
        # Make sure we have a copy to avoid modifying the original
        img_copy = pil_image.copy()
        
        # Debug print to verify image format and size
        print(f"Displaying image: mode={img_copy.mode}, size={img_copy.size}")
        
        # Convert PIL image to QPixmap at original size
        if img_copy.mode == "RGB":
            # RGB mode - direct conversion
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width * 3, 
                QImage.Format.Format_RGB888
            )
        elif img_copy.mode == "L":
            # Grayscale mode - direct conversion
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data,
                img_copy.width,
                img_copy.height,
                img_copy.width,
                QImage.Format.Format_Grayscale8
            )
        else:
            # For all other modes (including RGBA) convert to RGB
            rgb_image = img_copy.convert("RGB")
            img_data = rgb_image.tobytes()
            q_image = QImage(
                img_data,
                rgb_image.width,
                rgb_image.height,
                rgb_image.width * 3,
                QImage.Format.Format_RGB888
            )
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Set the image in the appropriate viewer based on current tab
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0:
            self.image_viewer.set_image(pixmap)
        elif current_tab_index == 1 and self.has_camera_tab:
            self.camera_viewer.set_image(pixmap)
    
    def floyd_steinberg_numpy(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
        
        if self.rgb_mode.isChecked():
            mode = 'RGB'
        else:
            mode = 'L'
        
        if pixel_scale == 1:
            # Convert PIL to numpy array once
            arr = np.array(pil_img.convert(mode), dtype=np.float32)
            # Process the array directly
            arr = fs_dither(arr, mode, threshold)    
            
            # Only convert back to PIL at the end
            return Image.fromarray(arr)
        else:
            # Use the optimized downscale-dither-upscale pipeline
            # Convert PIL to numpy array
            arr = np.array(pil_img.convert(mode))
            
            # Apply the full pipeline
            result = downscale_dither_upscale(arr, threshold, pixel_scale, mode)
            
            # Convert result back to PIL Image
            if mode == 'RGB':
                return Image.fromarray(result, 'RGB')
            else:
                return Image.fromarray(result, 'L')
    
    def simple_threshold(self, pil_img, threshold=128, pixel_scale=1):
        """Apply simple thresholding dithering"""
        # Convert to numpy array
        if self.rgb_mode.isChecked():
            mode = 'RGB'
            arr = np.array(pil_img.convert(mode))
        else:
            mode = 'L'
            arr = np.array(pil_img.convert(mode))
        
        # Choose implementation based on pixel scale
        if pixel_scale == 1:
            # Direct threshold for 1:1 mapping
            if mode == 'RGB':
                result = simple_threshold_rgb_ps1(arr, threshold)
            else:
                result = np.where(arr < threshold, 0, 255).astype(np.uint8)
        else:
            # Block-based threshold for larger pixels
            h, w = arr.shape[:2]
            result = simple_threshold_dither(arr, mode, pixel_scale, w, h, threshold)
        
        # Convert back to PIL
        if mode == 'RGB':
            return Image.fromarray(result, 'RGB')
        else:
            return Image.fromarray(result, 'L')
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if self.is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Initialize and start the camera"""
        # Don't allow starting if already running
        if self.is_camera_active or self.camera_thread is not None:
            print("Camera already active")
            return
            
        # Check if it's been less than 3 seconds since the last camera error
        current_time = time.time()
        if current_time - self._last_camera_error_time < 3.0:
            print("Blocking camera start - too soon after error")
            return
            
        print("Starting camera...")
        
        try:
            # Import camera thread class here for better error handling
            from classes.camera_capture_thread import CameraCaptureThread
            from classes.frame_processing_thread import FrameProcessingThread
            
            # Update UI
            self.camera_button.setText("Stop Camera")
            self.screenshot_button.setEnabled(True)
            
            # Initialize frame queue 
            self.frame_queue = queue.Queue(maxsize=3)  # Small queue to prevent memory buildup
            
            # Initialize and start camera thread
            self.camera_thread = CameraCaptureThread(self)
            self.camera_thread.frameProcessed.connect(self.on_frame_processed)
            self.camera_thread.start()
            
            # Initialize and start processing thread
            self.processing_thread = FrameProcessingThread(self.frame_queue, self)
            self.processing_thread.frameProcessed.connect(self.on_processed_frame)
            self.processing_thread.start()
            
            # Update state
            self.is_camera_active = True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            traceback.print_exc()
            
            # Cleanup any partially initialized threads
            self.stop_camera()
            
            # Update error time to prevent immediate restart attempts
            self._last_camera_error_time = time.time()
            
            # Update UI to reflect error
            self.camera_button.setText("Start Camera")
            
    def stop_camera(self):
        """Stop the camera and associated threads"""
        print("Stopping camera...")
        
        # Update UI first
        self.camera_button.setText("Start Camera")
        self.screenshot_button.setEnabled(False)
        
        # Stop processing thread
        if self.processing_thread is not None:
            print("Stopping processing thread...")
            try:
                self.processing_thread.stop()
                self.processing_thread.wait(1000)  # Wait up to 1 second
                
                # If still running after timeout, terminate forcefully
                if self.processing_thread.isRunning():
                    print("Processing thread did not stop cleanly, terminating")
                    self.processing_thread.terminate()
                
                self.processing_thread = None
            except Exception as e:
                print(f"Error stopping processing thread: {e}")
                traceback.print_exc()
        
        # Stop camera thread
        if self.camera_thread is not None:
            print("Stopping camera thread...")
            try:
                self.camera_thread.stop()
                self.camera_thread.wait(2000)  # Wait up to 2 seconds
                
                # If still running after timeout, terminate forcefully
                if self.camera_thread.isRunning():
                    print("Camera thread did not stop cleanly, terminating")
                    self.camera_thread.terminate()
                
                self.camera_thread = None
            except Exception as e:
                print(f"Error stopping camera thread: {e}")
                traceback.print_exc()
        
        # Clear queue
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
            
        # Clear current image
        if self.has_camera_tab:
            self.camera_viewer.set_image(None)
        
        # Update state
        self.is_camera_active = False
        print("Camera stopped")
    
    def take_screenshot(self):
        """Capture the current camera frame and save it as a still image"""
        if not self.is_camera_active:
            print("Camera not active")
            return
            
        # Save current frame as the "original" image
        current_frame = self.dithered_image
        if current_frame:
            print("Saving camera frame as screenshot")
            self.original_image = current_frame.copy()
            
            # Stop camera
            self.stop_camera()
            
            # Switch to file tab
            self.tabs.setCurrentIndex(0)
            
            # Apply dithering based on current settings and display
            self.update_dithering()
            
    def on_frame_processed(self, frame):
        """Receive frame directly from camera thread"""
        # Store as current dithered image so we can take screenshots
        self.dithered_image = frame
        
        # Display image directly
        if self.has_camera_tab and self.tabs.currentIndex() == 1:
            self.display_image(frame)
    
    def on_processed_frame(self, frame):
        """Receive frame from the processing thread"""
        # Store as current dithered image so we can take screenshots
        self.dithered_image = frame
        
        # Display image
        if self.has_camera_tab and self.tabs.currentIndex() == 1:
            self.display_image(frame)
    
    def update_ui(self):
        """Update UI elements on a timer"""
        # Update threshold value label
        self.threshold_value.setText(str(self.threshold_slider.value()))
        
        # Update contrast value label
        self.contrast_value.setText(f"{self.contrast_slider.value()}%")
        
        # Update pixel scale label
        self.scale_value.setText(f"{self.scale_slider.value()}x")
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        print("Application closing, cleaning up...")
        
        # Stop camera if running
        if self.is_camera_active:
            self.stop_camera()
            
        # Wait a bit to ensure resources are released
        time.sleep(0.5)
        
        # Accept the close event
        event.accept() 