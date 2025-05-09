import sys
import threading
import time
import queue
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QSlider, QFileDialog, QSplitter, QCheckBox,
                             QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent
from PIL import Image, ImageEnhance
import numpy as np
import os, config
from helper import fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither

# Try to import picamera only if available (for development on non-Pi platforms)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera module not available. Camera features will be disabled.")

class CameraCaptureThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue):
        super().__init__()
        self.is_running = False  # Start as not running
        self.frame_queue = frame_queue
        self.camera = None
        self.camera_initialized = False
        
    def run(self):
        print("Camera capture thread starting...")
        
        if not PICAMERA_AVAILABLE:
            print("Picamera2 not available, thread exiting")
            return
        
        self.is_running = True
        
        try:
            # Initialize camera
            self.camera = Picamera2()
            
            # Configure camera for preview/video streaming
            preview_config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
            )
            self.camera.configure(preview_config)
            
            # Start camera
            print("Starting camera...")
            self.camera.start()
            
            # Wait a moment for camera to initialize
            time.sleep(1.0)
            self.camera_initialized = True
            print("Camera initialized successfully")
            
            # Main capture loop
            frames_captured = 0
            last_report_time = time.time()
            
            while self.is_running:
                if not self.camera_initialized:
                    print("Camera not properly initialized, stopping thread")
                    break
                    
                try:
                    # Capture frame directly as numpy array
                    frame = self.camera.capture_array()
                    frames_captured += 1
                    
                    # Report FPS every 5 seconds
                    current_time = time.time()
                    if current_time - last_report_time > 5.0:
                        fps = frames_captured / (current_time - last_report_time)
                        print(f"Camera capture rate: {fps:.1f} FPS")
                        frames_captured = 0
                        last_report_time = current_time
                    
                    # Ensure the frame is in RGB format
                    if frame is not None and len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # If RGBA format
                            frame = frame[:, :, :3]  # Convert to RGB by removing alpha
                        
                        # Add captured frame to the queue
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame.copy())  # Make a copy to avoid reference issues
                    else:
                        print(f"Invalid frame format: {frame.shape if frame is not None else None}")
                        
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    # Don't break the loop for occasional errors
                    
                # Sleep to maintain frame rate
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            print(f"Critical error in camera capture thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always clean up camera resources
            self.stop_camera()
            print("Camera capture thread finished")
    
    def stop_camera(self):
        """Clean up camera resources safely"""
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                print("Stopping camera device...")
                self.camera.stop()
                print("Camera stopped successfully")
            except Exception as e:
                print(f"Error stopping camera: {e}")
            
            # Clear reference
            self.camera = None
            self.camera_initialized = False

    def stop(self):
        """Stop the thread safely"""
        print("Requesting camera thread to stop...")
        self.is_running = False
        self.stop_camera()  # Stop the camera right away

class FrameProcessingThread(QThread):
    frameProcessed = pyqtSignal(Image.Image)

    def __init__(self, frame_queue, app_instance):
        super().__init__()
        self.is_running = False  # Start as not running
        self.frame_queue = frame_queue
        self.app = app_instance  # Reference to the main app for dithering settings
        print("Processing thread initialized")

    def run(self):
        print("Frame processing thread starting...")
        self.is_running = True
        
        # Performance tracking
        frames_processed = 0
        last_report_time = time.time()
        
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    # Get frame from the queue without blocking (to check is_running state more often)
                    frame = self.frame_queue.get(block=False)
                    frames_processed += 1
                    
                    # Report performance periodically
                    current_time = time.time()
                    if current_time - last_report_time > 5.0:
                        fps = frames_processed / (current_time - last_report_time)
                        print(f"Processing rate: {fps:.1f} FPS")
                        frames_processed = 0
                        last_report_time = current_time
                    
                    # Make sure we have a proper RGB frame
                    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Convert numpy array to PIL Image
                        try:
                            pil_img = Image.fromarray(frame, mode='RGB')
                            
                            # Apply dithering using the main app's dithering functions
                            dithered_img = self.process_frame(pil_img)
                            if dithered_img:
                                # Emit processed frame - only if we're still running
                                if self.is_running:
                                    self.frameProcessed.emit(dithered_img)
                            else:
                                # If processing failed, emit the original image
                                if self.is_running:
                                    self.frameProcessed.emit(pil_img)
                        except Exception as e:
                            print(f"Error processing frame: {e}")
                    else:
                        # Skip this frame if format is wrong
                        if frame is None:
                            print("Received None frame")
                        else:
                            print(f"Skipping frame with format: {frame.shape}")
                else:
                    # No frames to process, sleep a bit to prevent CPU hogging
                    time.sleep(0.01)
                    
            except queue.Empty:
                # Frame queue is empty, just continue the loop
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
                # Don't exit the loop on error, keep trying

        print("Frame processing thread stopped")

    def process_frame(self, pil_img):
        """Apply dithering to a PIL image based on current app settings"""
        try:
            # Get current settings from main app
            alg = self.app.algorithm_combo.currentText()
            thr = self.app.threshold_slider.value()
            contrast_factor = self.app.contrast_slider.value() / 100.0
            pixel_s = self.app.scale_slider.value()
            
            # Apply contrast if needed
            image_to_dither = pil_img
            if abs(contrast_factor - 1.0) > 0.01:
                try:
                    enhancer = ImageEnhance.Contrast(pil_img)
                    image_to_dither = enhancer.enhance(contrast_factor)
                except Exception as e:
                    print(f"Error applying contrast: {e}")
                    # Continue with original image
            
            # Apply selected dithering algorithm
            if alg == "Floyd-Steinberg":
                result = self.app.floyd_steinberg_numpy(image_to_dither, thr, pixel_s)
            elif alg == "Simple Threshold":
                result = self.app.simple_threshold(image_to_dither, thr, pixel_s)
            else:
                result = image_to_dither  # Fallback
                
            return result
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def stop(self):
        """Stop the processing thread safely"""
        print("Requesting processing thread to stop...")
        self.is_running = False

class ImageViewer(QScrollArea):
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent; border: 1px solid #ddd;")
        
        self.setWidget(self.image_label)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

    def _apply_zoom_to_display(self):
        """Scales original_pixmap by zoom_factor and updates the label."""
        if not self.original_pixmap:
            self.image_label.clear()
            return

        new_width = int(self.original_pixmap.width() * self.zoom_factor)
        new_height = int(self.original_pixmap.height() * self.zoom_factor)
        
        scaled_pixmap = self.original_pixmap.scaled(
            new_width, new_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        
    def set_image(self, pixmap):
        if pixmap:
            self.original_pixmap = pixmap.copy()
            # Apply the current internal zoom_factor to the new pixmap.
            # Do not change self.zoom_factor here.
            self._apply_zoom_to_display()
        else: # Image is being cleared
            self.original_pixmap = None
            self._apply_zoom_to_display() # This will clear the label
            # If image is cleared, reset internal zoom and notify to reset slider.
            if abs(self.zoom_factor - 1.0) > 0.001:
                self.zoom_factor = 1.0
                self.zoom_changed.emit(1.0)
            elif self.image_label.pixmap() is None: # Ensure emit if already 1.0 but no image
                 self.zoom_changed.emit(1.0)

    def set_zoom_level(self, level: float):
        new_zoom_factor = max(self.min_zoom, min(self.max_zoom, level))

        # Only update if the factor actually changes or if there's no image (to set initial zoom state)
        if abs(self.zoom_factor - new_zoom_factor) > 0.001 or (not self.original_pixmap and self.image_label.pixmap() is None):
            self.zoom_factor = new_zoom_factor
            self._apply_zoom_to_display()
            self.zoom_changed.emit(self.zoom_factor)
            
    def wheelEvent(self, event: QWheelEvent):
        # if not self.original_pixmap: # Covered by set_zoom_level not applying if no pixmap
        #     super().wheelEvent(event)
        #     return

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle_delta = event.angleDelta().y()
            proposed_zoom_factor = self.zoom_factor
            
            if angle_delta > 0:
                proposed_zoom_factor *= 1.1
            else:
                proposed_zoom_factor *= 0.9
            
            self.set_zoom_level(proposed_zoom_factor)
            event.accept()
        else:
            super().wheelEvent(event)

class DitherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1200, 800)
        
        self.original_image = None
        self.dithered_image = None
        self.showing_original = False # Show dithered version first by default
        
        # Initialize camera mode as disabled by default
        self.camera_mode_active = False
        self.frame_queue = queue.Queue(maxsize=10)  # For camera frames
        self.capture_thread = None
        self.processing_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main splitter widget
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # Left panel (80%) - Image display
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        
        # Image viewer with scrollbars
        self.image_viewer = ImageViewer()
        self.image_layout.addWidget(self.image_viewer, 1)
        
        # "Switch to Original Image" button - will be created and styled later for control_panel
        self.toggle_button = QPushButton("Switch to Original Image")
        self.toggle_button.clicked.connect(self.toggle_image_display)
        self.toggle_button.setEnabled(False)
        self.toggle_button.setMinimumWidth(200)
        self.toggle_button.setFixedHeight(40)
        
        # Note: toggle_button is NOT added to image_layout here anymore
        
        # Right panel (20%) - Controls
        self.control_panel = QWidget()
        self.control_panel.setObjectName("controlPanelBackground")

        # Assuming 'control_panel_background.png' is in the same directory as main.py
       
        # More specific stylesheet for control_panel and general styles for its children
        self.control_panel.setStyleSheet(f"""
            #controlPanelBackground {{
                border-image: url({os.path.join(os.path.dirname(__file__), "background.jpg")}) 0 0 0 0 stretch stretch;
            }}

            /* Styles for children of controlPanelBackground for readability */
            #controlPanelBackground QLabel {{
                background-color: transparent;
                color: white;
            }}
            #controlPanelBackground QCheckBox {{
                background-color: transparent;
                color: white;
            }}
            #controlPanelBackground QPushButton {{
                border: 1px solid #BBB;
                border-radius: 15px; 
                padding: 5px;
                background-color: rgba(200, 200, 200, 0.3); /* Semi-transparent background */
                color: white;
            }}
            #controlPanelBackground QPushButton:hover {{
                background-color: rgba(220, 220, 220, 0.5);
            }}
            #controlPanelBackground QPushButton:pressed {{
                background-color: rgba(180, 180, 180, 0.5);
            }}
        """)

        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.control_layout.setContentsMargins(10, 20, 10, 20)
        self.control_layout.setSpacing(10)
        
        control_title = QLabel("Controls")
        # control_title style will be inherited or can be set specifically if needed
        self.control_layout.addWidget(control_title)
        
        button_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setMinimumWidth(120) # Style will come from panel stylesheet
        button_layout.addWidget(self.open_button)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        self.save_button.setMinimumWidth(120) # Style will come from panel stylesheet
        button_layout.addWidget(self.save_button)
        
        button_container = QWidget() # This container itself won't get the background image
        button_container.setLayout(button_layout)
        self.control_layout.addWidget(button_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Camera buttons
        camera_button_layout = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera_mode)
        self.camera_button.setMinimumWidth(120)
        
        # Only enable camera button if picamera is available
        self.camera_button.setEnabled(PICAMERA_AVAILABLE)
        
        camera_button_layout.addWidget(self.camera_button)
        
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setMinimumWidth(120)
        self.capture_button.setEnabled(False)  # Disabled until camera starts
        camera_button_layout.addWidget(self.capture_button)
        
        camera_button_container = QWidget()
        camera_button_container.setLayout(camera_button_layout)
        self.control_layout.addWidget(camera_button_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add horizontal divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        self.control_layout.addWidget(divider)
        
        # Zoom Slider
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 500)  # Represents 10% to 500%
        self.zoom_slider.setValue(100)       # Default 100% (1.0x zoom)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.handle_zoom_slider_change)
        self.zoom_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.zoom_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Algorithm selector 
        self.algorithm_label = QLabel("Algorithm:")
        self.algorithm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.algorithm_label)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Floyd-Steinberg", "Simple Threshold"])
        self.algorithm_combo.currentIndexChanged.connect(self.algorithm_changed)
        self.algorithm_combo.setMinimumWidth(180)
        self.control_layout.addWidget(self.algorithm_combo, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Threshold slider
        self.threshold_label = QLabel("Threshold: 128")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 254)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.threshold_changed)
        self.threshold_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.threshold_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Contrast slider
        self.contrast_label = QLabel("Contrast: 1.0")
        self.contrast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.contrast_label)
        
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 500)  # 0.1 to 5.0 scaled by 100
        self.contrast_slider.setValue(100)      # 1.0 * 100
        self.contrast_slider.valueChanged.connect(self.contrast_changed)
        self.contrast_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.contrast_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Pixel Scale slider
        self.scale_label = QLabel("Pixel Scale: 1")
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.scale_label)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(1, 8)
        self.scale_slider.setValue(1)
        self.scale_slider.valueChanged.connect(self.scale_changed)
        self.scale_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.scale_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add another divider
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setFrameShadow(QFrame.Shadow.Sunken)
        self.control_layout.addWidget(divider2)
        
        # Auto-render and RGB checkboxes
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        
        self.auto_render = QCheckBox("Auto-Render")
        self.auto_render.setChecked(True)
        checkbox_layout.addWidget(self.auto_render)
        
        self.rgb_mode = QCheckBox("Color")
        self.rgb_mode.setChecked(True)
        self.rgb_mode.stateChanged.connect(self.rgb_changed)
        checkbox_layout.addWidget(self.rgb_mode)
        
        self.control_layout.addWidget(checkbox_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Create Apply Dither button (will be added to its own group later)
        self.apply_button = QPushButton("Apply Dither")
        self.apply_button.clicked.connect(self.apply_dither)
        self.apply_button.setMinimumWidth(200)
        self.apply_button.setFixedHeight(40)
        # Individual style removed, will be handled by control_panel stylesheet

        # Add a stretch to push the following button group towards the bottom
        self.control_layout.addStretch(1)

        # Vertical button group for Toggle and Apply buttons
        action_button_group_widget = QWidget()
        action_button_group_layout = QVBoxLayout(action_button_group_widget)
        action_button_group_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_button_group_layout.setSpacing(10) # Spacing between toggle and apply
        
        action_button_group_layout.addWidget(self.toggle_button) # Switch button on top
        action_button_group_layout.addWidget(self.apply_button)  # Apply button below
        
        self.control_layout.addWidget(action_button_group_widget, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Final stretch to take up any remaining space at the very bottom
        self.control_layout.addStretch(0) 
        
        # Add both panels to splitter with 80/20 ratio
        self.splitter.addWidget(self.image_panel)
        self.splitter.addWidget(self.control_panel)
        
        # Calculate the actual 80/20 split based on the window size
        total_width = self.width()
        self.splitter.setSizes([int(total_width * 0.8), int(total_width * 0.2)])
        
        # Connect ImageViewer zoom changes to update slider/label
        self.image_viewer.zoom_changed.connect(self.update_controls_from_zoom_factor)
    
    def toggle_camera_mode(self):
        if self.camera_mode_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera capture and processing, ensuring proper synchronization"""
        if not PICAMERA_AVAILABLE:
            print("Camera not available")
            return
            
        # If camera is already active, stop it first to ensure clean state
        if self.camera_mode_active:
            print("Camera already active, stopping first")
            self.stop_camera()
            # Wait a moment to ensure everything is stopped
            time.sleep(0.5)
        
        print("Starting camera...")
        
        # Clear queue to prevent old frames
        while not self.frame_queue.empty():
            self.frame_queue.get()
            
        # Create and start threads if they don't exist
        if not self.capture_thread or not self.capture_thread.isRunning():
            if self.capture_thread:
                del self.capture_thread  # Delete old thread if it exists but isn't running
            self.capture_thread = CameraCaptureThread(self.frame_queue)
            
        if not self.processing_thread or not self.processing_thread.isRunning():
            if self.processing_thread:
                del self.processing_thread  # Delete old thread if it exists but isn't running
            self.processing_thread = FrameProcessingThread(self.frame_queue, self)
            self.processing_thread.frameProcessed.connect(self.update_camera_frame)
            
        # Start threads
        if not self.capture_thread.isRunning():
            self.capture_thread.start()
            print("Camera capture thread started")
            
        if not self.processing_thread.isRunning():
            self.processing_thread.start()
            print("Frame processing thread started")
        
        # Wait briefly to ensure threads are running
        time.sleep(0.2)
        
        # Check if threads actually started
        if not self.capture_thread.isRunning() or not self.processing_thread.isRunning():
            print("Failed to start camera threads")
            self.stop_camera()
            return
            
        # Update UI
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.open_button.setEnabled(False)  # Disable file open during camera mode
        self.camera_mode_active = True
        
        # Clear current image
        self.showing_original = False
        self.original_image = None
        self.dithered_image = None
        self.toggle_button.setEnabled(False)
        
        print("Camera mode started successfully")
    
    def stop_camera(self):
        """Stop camera capture and processing, ensuring proper synchronization"""
        print("Stopping camera...")
        self.camera_mode_active = False  # Set this first to prevent new frames processing
        
        # Stop threads with timeout handling
        thread_stop_timeout = 3.0  # seconds
        stop_successful = True
        
        # Stop capture thread
        if self.capture_thread and self.capture_thread.isRunning():
            print("Stopping capture thread...")
            self.capture_thread.stop()
            
            # Wait with timeout for thread to finish
            start_time = time.time()
            while self.capture_thread.isRunning() and (time.time() - start_time) < thread_stop_timeout:
                time.sleep(0.1)
                
            if self.capture_thread.isRunning():
                print("Warning: Capture thread did not stop within timeout")
                # Thread is still running - this is not good, but we've done what we can
                stop_successful = False
            else:
                print("Capture thread stopped successfully")
        
        # Stop processing thread
        if self.processing_thread and self.processing_thread.isRunning():
            print("Stopping processing thread...")
            self.processing_thread.stop()
            
            # Wait with timeout for thread to finish
            start_time = time.time()
            while self.processing_thread.isRunning() and (time.time() - start_time) < thread_stop_timeout:
                time.sleep(0.1)
                
            if self.processing_thread.isRunning():
                print("Warning: Processing thread did not stop within timeout")
                stop_successful = False
            else:
                print("Processing thread stopped successfully")
        
        # Clear queue
        while not self.frame_queue.empty():
            self.frame_queue.get()
            
        # Update UI
        self.camera_button.setText("Start Camera")
        self.capture_button.setEnabled(False)
        self.open_button.setEnabled(True)  # Re-enable file open
        
        # Clear image display
        self.image_viewer.set_image(None)
        
        if stop_successful:
            print("Camera stopped successfully")
        else:
            print("Camera stop had issues - some threads may still be running")
            
        # Force application to process events to update UI
        QApplication.processEvents()
    
    def capture_frame(self):
        """Capture current frame and save it as the original image"""
        if not self.camera_mode_active:
            print("Camera not active, cannot capture frame")
            return
            
        if self.frame_queue.empty():
            print("No frames in queue")
            return
            
        try:
            # Get the most recent frame from the queue
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
            if frame is not None:
                # Convert to PIL image and store
                print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
                self.original_image = Image.fromarray(frame)
                print(f"Converted to PIL image: mode={self.original_image.mode}, size={self.original_image.size}")
                
                # Stop camera and apply dithering
                self.stop_camera()
                self.apply_dither()
                
                # Enable UI elements for captured image
                if self.dithered_image:
                    self.save_button.setEnabled(True)
                    self.toggle_button.setEnabled(True)
                    print("Dithering applied successfully")
                else:
                    print("Failed to apply dithering to captured frame")
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            import traceback
            traceback.print_exc()
    
    def update_camera_frame(self, dithered_img):
        """Update the UI with the processed camera frame"""
        if not self.camera_mode_active:
            return
            
        # Display the dithered frame
        self.display_image(dithered_img)
        # Also keep a reference to it as the dithered image so it can be saved
        self.dithered_image = dithered_img
        # Enable save button for camera frames
        self.save_button.setEnabled(True)
    
    def toggle_image_display(self):
        if not self.original_image or not self.dithered_image:
            return
        
        self.showing_original = not self.showing_original
        
        if self.showing_original:
            self.display_image(self.original_image)
            self.toggle_button.setText("Switch to Dithered Image")
        else:
            self.display_image(self.dithered_image)
            self.toggle_button.setText("Switch to Original Image")
    
    def open_image(self):
        # Ensure camera is stopped first
        if self.camera_mode_active:
            self.stop_camera()
            
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        
        self.open_path = path
        try:
            self.original_image = Image.open(path)
            # self.showing_original is already False from __init__
            
            self.apply_dither() # Attempt to dither and display immediately
            self.image_viewer.set_zoom_level(1.0) # Reset zoom for the new image view

            if self.dithered_image:
                # Dithering was successful, dithered image is shown
                self.toggle_button.setText("Switch to Original Image")
                self.save_button.setEnabled(True)
                self.toggle_button.setEnabled(True) # Original also exists
            else:
                # Initial dithering failed, fall back to showing original
                self.showing_original = True # Update state to reflect original is shown
                self.display_image(self.original_image) # Explicitly display original
                self.toggle_button.setText("Switch to Dithered Image")
                self.save_button.setEnabled(False)
                self.toggle_button.setEnabled(False) # No dithered image to switch to
                
        except Exception as e:
            print(f"Error opening image: {e}")
            self.original_image = None
            self.dithered_image = None
            self.image_viewer.set_image(None) 
            self.save_button.setEnabled(False)
            self.toggle_button.setEnabled(False)
            self.showing_original = False # Reset for next attempt
    
    def display_image(self, pil_image):
        if not pil_image:
            self.image_viewer.set_image(None) # set_image(None) handles zoom reset
            return
        
        # Make sure we have a copy to avoid modifying the original
        img_copy = pil_image.copy()
        
        # Debug print to verify image format and size
        print(f"Displaying image: mode={img_copy.mode}, size={img_copy.size}")
        
        # Convert PIL image to QPixmap at original size
        if img_copy.mode == "RGBA":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width * 4, # RGBA has 4 bytes per pixel
                QImage.Format.Format_RGBA8888
            )
        elif img_copy.mode == "RGB":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width * 3, 
                QImage.Format.Format_RGB888
            )
        elif img_copy.mode == "L":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width, 
                QImage.Format.Format_Grayscale8
            )
        elif img_copy.mode == "P": # Palette-based images
            img_copy = img_copy.convert("RGBA") # Convert to RGBA to handle transparency properly
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data,
                img_copy.width,
                img_copy.height,
                img_copy.width * 4,
                QImage.Format.Format_RGBA8888
            )
        else: # For other modes, try converting to RGBA
            try:
                img_copy = img_copy.convert("RGBA")
                img_data = img_copy.tobytes()
                q_image = QImage(
                    img_data, 
                    img_copy.width, 
                    img_copy.height, 
                    img_copy.width * 4, 
                    QImage.Format.Format_RGBA8888
                )
            except Exception as e:
                print(f"Could not convert image mode {img_copy.mode} to RGBA: {e}")
                self.image_viewer.set_image(None)
                return
        
        # Debug print to verify QImage creation
        print(f"Created QImage: {q_image.width()}x{q_image.height()}, format={q_image.format()}")
        
        pixmap = QPixmap.fromImage(q_image)
        if pixmap.isNull():
            print("Error: Created QPixmap is null")
        else:
            print(f"Created QPixmap: {pixmap.width()}x{pixmap.height()}")
            
        self.image_viewer.set_image(pixmap)
    
    def threshold_changed(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def contrast_changed(self, value):
        contrast = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast:.1f}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def scale_changed(self, value):
        self.scale_label.setText(f"Pixel Scale: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def algorithm_changed(self):
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def rgb_changed(self):
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def apply_dither(self):
        if not self.original_image:
            return
        
        alg = self.algorithm_combo.currentText()
        thr = self.threshold_slider.value()
        contrast_factor = self.contrast_slider.value() / 100.0
        pixel_s = self.scale_slider.value()
        
        print(f"Applying {alg} with threshold {thr}, contrast {contrast_factor:.2f}, scale {pixel_s}")
        
        image_to_dither = self.original_image
        if abs(contrast_factor - 1.0) > 0.01:
            try:
                enhancer = ImageEnhance.Contrast(self.original_image)
                image_to_dither = enhancer.enhance(contrast_factor)
                print("Applied contrast adjustment.")
            except Exception as e:
                print(f"Error applying contrast: {e}")
                image_to_dither = self.original_image
        
        if alg == "Floyd-Steinberg":
            self.dithered_image = self.floyd_steinberg_numpy(image_to_dither, thr, pixel_s)
        elif alg == "Simple Threshold":
            self.dithered_image = self.simple_threshold(image_to_dither, thr, pixel_s)
        else:
            self.dithered_image = None
        
        # Update toggle button state
        if self.dithered_image:
            self.toggle_button.setEnabled(True)
            
            # If not showing original, update display with new dithered image
            if not self.showing_original:
                self.display_image(self.dithered_image)
    
    def save_image(self):
        # Check if we have a dithered image to save (either from file or camera)
        if not self.dithered_image:
            print("No image to save")
            return
        
        try:
            # Debug info
            print(f"Attempting to save image: {type(self.dithered_image)}, "
                  f"mode={self.dithered_image.mode if hasattr(self.dithered_image, 'mode') else 'unknown'}")
            
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*.*)"
            )
            if path:
                print(f"Saving to path: {path}")
                # Ensure the image is in a format that can be saved
                if hasattr(self.dithered_image, 'save'):
                    self.dithered_image.save(path)
                    print(f"Image saved to {path}")
                else:
                    print(f"Error: dithered_image does not have save method: {type(self.dithered_image)}")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def floyd_steinberg_numpy(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
        
        if self.rgb_mode.isChecked():
            type = 'RGB'
        else:
            type = 'L'
        
        if pixel_scale == 1:
            img = pil_img.convert(type)
            arr = np.array(img, dtype=np.float32)
            print("type", type)
            arr = fs_dither(arr, type, threshold)    
            
            return Image.fromarray(arr)
        else:
            orig_w, orig_h = pil_img.size
            small_w = max(1, orig_w // pixel_scale)
            small_h = max(1, orig_h // pixel_scale)
            
            print(f"Downscaling to {small_w}x{small_h}")
            small_img = pil_img.resize((small_w, small_h), Image.Resampling.BOX)
            
            print("Dithering downscaled image...")
            dithered_small_img = self.floyd_steinberg_numpy(small_img, threshold, pixel_scale=1)
            
            print(f"Upscaling back to {orig_w}x{orig_h}")
            dithered_large_img = dithered_small_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
            return dithered_large_img
    
    def simple_threshold(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
        
        if self.rgb_mode.isChecked():
            type = 'RGB'
        else:
            type = 'L'
        img_gray = pil_img.convert(type)
        
        if pixel_scale == 1:
            if type == 'RGB':
                # Process each band separately for RGB using NumPy
                img_array = np.array(img_gray)
                result = simple_threshold_rgb_ps1(img_array, threshold)
                return Image.fromarray(result.astype(np.uint8))
            else:
                # For grayscale, use PIL's point for threshold (faster for single channel)
                return img_gray.point(lambda p: 0 if p < threshold else 255)
        else:
            # For larger pixel scales, we need the block-based approach
            orig_w, orig_h = img_gray.size
            
            # Convert to numpy array
            img_array = np.array(img_gray)
            
            # Create output array of same shape
            out_array = simple_threshold_dither(img_array, type, pixel_scale, orig_w, orig_h, threshold)
            
            return Image.fromarray(out_array)

    def handle_zoom_slider_change(self, value):
        zoom_level = value / 100.0  # Convert slider value (10-500) to factor (0.1-5.0)
        # Update label immediately for responsiveness, even if viewer clamps it
        self.zoom_label.setText(f"Zoom: {value}%") 
        self.image_viewer.set_zoom_level(zoom_level)

    def update_controls_from_zoom_factor(self, zoom_factor):
        slider_value = int(zoom_factor * 100)
        # Block signals from slider to prevent feedback loop if slider.setValue itself triggers valueChanged
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(slider_value)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"Zoom: {slider_value}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update splitter ratio when window is resized
        total_width = self.width()
        self.splitter.setSizes([int(total_width * 0.8), int(total_width * 0.2)])
        
    def closeEvent(self, event):
        """Called when the application is closing, ensures threads are stopped"""
        print("Application closing, cleaning up resources...")
        # Stop camera threads if active
        if self.camera_mode_active:
            self.stop_camera()
            
        # Just to be thorough, check if threads exist and make sure they're stopped
        if self.capture_thread and self.capture_thread.isRunning():
            print("Forcing capture thread to stop...")
            self.capture_thread.stop()
            self.capture_thread.wait(1000)  # Wait up to 1 second
            
        if self.processing_thread and self.processing_thread.isRunning():
            print("Forcing processing thread to stop...")
            self.processing_thread.stop()
            self.processing_thread.wait(1000)  # Wait up to 1 second
            
        # Accept the event and close the application
        print("Cleanup complete, closing application")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DitherApp()
    window.show()
    sys.exit(app.exec())
