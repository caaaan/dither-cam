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
import numpy as np
import os, config
from helper import (fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither, 
                   block_average_rgb, block_average_gray, nearest_upscale_rgb, 
                   nearest_upscale_gray, downscale_dither_upscale, bgr_to_rgb, 
                   optimized_pass_through)

# Global frame handling variables
# These are used across the application to manage frame flow
FRAME_BUFFER_ORIGINAL = None     # Original frame buffer (RGB format)
FRAME_BUFFER_PROCESSING = None   # Buffer used during processing steps
FRAME_BUFFER_OUTPUT = None       # Final frame after processing (to display)
FRAME_BUFFER_GRAYSCALE = None    # Grayscale version when needed
SHARED_ALGORITHM_BUFFER = None   # Shared buffer for all algorithms - prevents memory allocation lag

# Frame handling metrics
LAST_FRAME_TIME = 0              # Time when last frame was processed
LAST_FPS = 0                     # Last calculated FPS
FRAME_COUNT = 0                  # Frame counter for FPS calculation
FRAME_PROCESS_TIME = 0           # Time spent processing the last frame

# Try to import picamera only if available (for development on non-Pi platforms)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera module not available. Camera features will be disabled.")

class ImageViewer(QScrollArea):
    """Custom image viewer widget with zoom functionality"""
    
    # Signal to notify when zoom factor changes
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize main widget and layout
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        
        # Set the label as the widget for the scroll area
        self.setWidget(self.image_label)
        self.setWidgetResizable(True)
        
        # Initialize variables
        self.pixmap = None
        self.current_image = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # Setup scrollbar policy
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Setup appearance
        self.setFrameShape(QFrame.Shape.NoFrame)
        
    def set_image(self, image):
        """Set a new image to display (accepts NumPy arrays or QImage)"""
        if image is None:
            self.image_label.clear()
            self.pixmap = None
            self.current_image = None
            return
        
        # Convert image to QImage if it's a NumPy array
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            
            # Convert based on image type (grayscale or RGB)
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                # Grayscale image
                qimage = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
            else:
                # RGB image - assume image is already in RGB format (not BGR)
                bytes_per_line = width * 3
                qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        elif isinstance(image, QImage):
            qimage = image
        else:
            # Unsupported image type
            print(f"Unsupported image type: {type(image)}")
            return
        
        # Create pixmap from QImage
        self.pixmap = QPixmap.fromImage(qimage)
        self.current_image = image
        
        # Apply current zoom factor
        self.update_view()
        
    def update_view(self):
        """Update the view with current zoom factor"""
        if self.pixmap is None:
            return
            
        # Calculate scaled size
        scaled_size = self.pixmap.size() * self.zoom_factor
        
        # Create scaled pixmap and set it to the label
        scaled_pixmap = self.pixmap.scaled(
            scaled_size, 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Set the scaled pixmap to the label
        self.image_label.setPixmap(scaled_pixmap)
        
        # Resize the label to fit the scaled image
        self.image_label.resize(scaled_pixmap.size())
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming"""
        if self.pixmap is None:
            # No image to zoom
            return
            
        # Get the amount of scrolling
        delta = event.angleDelta().y()
        
        # Calculate new zoom factor - faster zooming with larger steps
        zoom_speed = 0.002 if self.zoom_factor < 1.0 else 0.005
        new_zoom = self.zoom_factor + (delta * zoom_speed)
        
        # Enforce zoom limits
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        # Only update if zoom factor actually changed
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.update_view()
            
            # Emit signal about zoom change
            self.zoom_changed.emit(self.zoom_factor)
            
    def set_zoom(self, zoom_factor):
        """Set zoom factor directly"""
        if zoom_factor != self.zoom_factor:
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, zoom_factor))
            self.update_view()

def bgr_to_rgb_array(array):
    """Convert BGR array to RGB using simple NumPy slice operation"""
    if array is None or len(array.shape) < 3 or array.shape[2] < 3:
        return array
    return array[:, :, ::-1].copy()

def rgb_to_bgr_array(array):
    """Convert RGB array to BGR using simple NumPy slice operation"""
    # Same operation as BGR to RGB but kept separate for code clarity
    if array is None or len(array.shape) < 3 or array.shape[2] < 3:
        return array
    return array[:, :, ::-1].copy()

class CameraCaptureThread(QThread):
    frameProcessed = pyqtSignal(np.ndarray)  # Emit processed frames directly

    def __init__(self, app_instance):
        super().__init__()
        self.is_running = False  # Start as not running
        self.app = app_instance  # Reference to main app for processing settings
        self.camera = None
        self.camera_initialized = False
        self.camera_lock = threading.Lock()  # Add lock for thread safety
        
        # Use global buffer instead of local buffer
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        
        # Simple timing control - adjusted for 30 FPS target
        self.last_capture_time = 0
        self.min_capture_interval = 0.033  # 30fps target (33ms)
        
        # Performance tracking
        self.frames_processed = 0
        self.last_report_time = time.time()
        
        # Resolution control
        self.current_pixel_scale = 1
        self.base_width = 640  # Base resolution width
        self.base_height = 480  # Base resolution height
        
        # Dynamic frame skipping for CPU load management
        self.frames_to_skip = 0
        self.skip_counter = 0
        self.target_cpu_percent = 70  # Increased target CPU usage for better performance
        self.last_cpu_check = time.time()
        self.cpu_check_interval = 1.0  # Check CPU usage every second
        self.cpu_usage_history = []  # Initialize history list
        self.avg_process_time = 0.033  # Initial estimate: 33ms per frame
        
        # Size-based optimization
        self.is_small_resolution = False  # Will be set based on frame size
        self.small_frame_skip_factor = 1.5  # Skip more frames for small resolutions
        
    def reconfigure_resolution(self, pixel_scale):
        """Dynamically reconfigure the camera resolution based on pixel scale"""
        if not self.camera_initialized or self.camera is None:
            return False
        
        if pixel_scale == self.current_pixel_scale:
            # No change needed
            return True
            
        try:
            print(f"Reconfiguring camera resolution for pixel scale: {pixel_scale}")
            
            # Calculate new resolution
            new_width = max(320, self.base_width // pixel_scale)
            new_height = max(240, self.base_height // pixel_scale)
            
            # Make width and height even numbers (required by some cameras)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            print(f"New camera resolution: {new_width}x{new_height}")
            
            with self.camera_lock:
                # Stop camera before reconfiguring
                self.camera.stop()
                time.sleep(0.5)
                
                # Create new configuration with updated resolution
                preview_config = self.camera.create_still_configuration(
                    main={"size": (new_width, new_height), "format": "RGB888"}
                )
                
                # Apply new configuration
                self.camera.configure(preview_config)
                time.sleep(0.5)
                
                # Restart the camera
                self.camera.start()
                time.sleep(1.0)
                
                # Test capture to verify it's working
                test_frame = self.camera.capture_array()
                if test_frame is None:
                    raise RuntimeError("Failed to capture test frame after reconfiguration")
                
                print(f"Camera reconfigured successfully. Frame shape: {test_frame.shape}")
                self.current_pixel_scale = pixel_scale
                
                # Clear buffer to force recreation with new size
                FRAME_BUFFER_ORIGINAL = None
                
                return True
                
        except Exception as e:
            print(f"Error reconfiguring camera: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to restore camera to running state
            try:
                self.camera.start()
            except:
                pass
                
            return False
        
    def get_cpu_usage(self):
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except:
            return 0  # Return 0 if psutil is not available
            
    def update_dynamic_frame_skip(self):
        """Adjust frames to skip based on current CPU usage to maintain target CPU usage"""
        now = time.time()
        if now - self.last_cpu_check < self.cpu_check_interval:
            return
            
        self.last_cpu_check = now
        current_cpu = self.get_cpu_usage()
        self.cpu_usage_history.append(current_cpu)
        
        # Keep only the last 5 measurements
        if len(self.cpu_usage_history) > 5:
            self.cpu_usage_history.pop(0)
            
        # Calculate average CPU usage
        avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
        
        # Check if we're dealing with a small resolution
        if hasattr(self, 'frame_buffer') and self.frame_buffer is not None:
            height, width = self.frame_buffer.shape[:2]
            self.is_small_resolution = (height * width < 150000)  # Approx 400x400 or smaller
        
        # Base skip adjustment
        skip_adjustment = 0
        
        # Adjust frames to skip based on CPU usage
        if avg_cpu > self.target_cpu_percent + 5:  # CPU usage too high
            skip_adjustment = 1  # Increase skip
        elif avg_cpu < self.target_cpu_percent - 5 and self.frames_to_skip > 0:  # CPU usage too low
            skip_adjustment = -1  # Decrease skip
            
        # Apply skip adjustment with a factor for small resolutions
        if self.is_small_resolution and skip_adjustment > 0:
            # More aggressive skipping for small resolutions
            skip_adjustment = int(skip_adjustment * self.small_frame_skip_factor)
            
        # Apply the adjustment
        self.frames_to_skip = max(0, min(15, self.frames_to_skip + skip_adjustment))
            
        # If process time is very low, don't skip frames
        if self.avg_process_time < 0.01 and not self.is_small_resolution:  # Less than 10ms
            self.frames_to_skip = 0
            
        if self.frames_to_skip > 0 and len(self.cpu_usage_history) >= 3:
            print(f"CPU: {avg_cpu:.1f}%, skipping {self.frames_to_skip} frames, process time: {self.avg_process_time*1000:.1f}ms, small res: {self.is_small_resolution}")
        
    def run(self):
        print("Camera thread starting...")
        
        if not PICAMERA_AVAILABLE:
            print("Picamera2 not available, thread exiting")
            return
        
        self.is_running = True
        
        # Access global frame buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        global FRAME_BUFFER_GRAYSCALE
        global LAST_FRAME_TIME
        global FRAME_COUNT
        global LAST_FPS
        global FRAME_PROCESS_TIME
        
        try:
            # Try to initialize camera with a simpler approach
            max_attempts = 3
            self.camera = None
            self.camera_initialized = False
                
            for attempt in range(max_attempts):
                try:
                    print(f"Camera initialization attempt {attempt+1}/{max_attempts}")
                    
                    # First attempt: simplest initialization
                    with self.camera_lock:
                        self.camera = Picamera2()
                        
                        # Wait before configuring
                        time.sleep(2.0)
                        
                        # Get current pixel scale
                        pixel_scale = self.app.scale_slider.value()
                        self.current_pixel_scale = pixel_scale
                        
                        # Calculate resolution based on pixel scale
                        width = max(320, self.base_width // pixel_scale)
                        height = max(240, self.base_height // pixel_scale)
                        
                        # Make width and height even numbers
                        width = width - (width % 2)
                        height = height - (height % 2)
                        
                        print(f"Initial camera resolution: {width}x{height}")
                        
                        # Use the simplest configuration possible with resolution
                        preview_config = self.camera.create_still_configuration(
                            main={"size": (width, height), "format": "RGB888"}
                        )
                        
                        print(f"Using camera config: {preview_config}")
                        self.camera.configure(preview_config)
                        
                        # Wait before starting
                        time.sleep(2.0)
                        
                        print("Starting camera...")
                        self.camera.start()
                        
                        # Wait for camera to initialize
                        time.sleep(3.0)
                        
                        # Add note about picamera2 color format
                        print("Note: picamera2 typically outputs frames in BGR format, will be converted to RGB")
                        
                        # Test by capturing one frame - if this succeeds, camera is working
                        test_frame = self.camera.capture_array()
                        if test_frame is not None:
                            print(f"Test frame captured successfully: {test_frame.shape}, data type: {test_frame.dtype}")
                            # Print the first few pixels to debug color format
                            if len(test_frame.shape) == 3 and test_frame.shape[2] >= 3:
                                print(f"First pixel RGB/BGR values: {test_frame[0,0,:]}")
                            self.camera_initialized = True
                            print("Camera initialized successfully")
                            break
                        else:
                            raise RuntimeError("Test frame capture returned None")
                            
                except Exception as e:
                    print(f"Camera initialization attempt {attempt+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Clean up resources before trying again
                    try:
                        if self.camera is not None:
                            self.camera.close()
                    except Exception:
                        pass
                        
                    self.camera = None
                    self.camera_initialized = False
                    
                    # Wait longer before next attempt
                    time.sleep(3.0)
                    
                    # If this was the last attempt, give up
                    if attempt == max_attempts - 1:
                        print("All camera initialization attempts failed")
                        self.is_running = False
                        return
            
            # Main capture loop - only reached if initialization succeeded
            frames_captured = 0
            last_report_time = time.time()
            self.last_capture_time = time.time()
                
            while self.is_running:
                if not self.camera_initialized or self.camera is None:
                    print("Camera not properly initialized, stopping thread")
                    break
                
                # Simple rate control - limit capture rate to avoid overloading CPU
                current_time = time.time()
                time_since_last_capture = current_time - self.last_capture_time
                
                if time_since_last_capture < self.min_capture_interval:
                    # Sleep only if we need to wait a meaningful amount
                    sleep_time = self.min_capture_interval - time_since_last_capture
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                    continue
                
                # Measure frame processing time
                process_start = time.time()
                
                try:
                    # Check if we should still be running
                    if not self.is_running:
                        break
                        
                    # Capture frame with minimal lock time
                    with self.camera_lock:
                        if self.camera is None:
                            print("Camera object is None, exiting capture loop")
                            break
                        frame = self.camera.capture_array()
                    
                    frames_captured += 1
                    
                    # Process valid frames
                    if frame is not None and len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # Convert RGBA to RGB if needed
                            frame = frame[:, :, :3]
                        
                        # Convert BGR to RGB for all processing paths
                        # Simple BGR to RGB conversion with NumPy - more efficient than loading OpenCV
                        frame = frame[:, :, ::-1].copy()  # Reverse the color channels
                        
                        # Use global buffer instead of thread-local buffer
                        if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_ORIGINAL.shape != frame.shape:
                            FRAME_BUFFER_ORIGINAL = np.empty_like(frame)
                        np.copyto(FRAME_BUFFER_ORIGINAL, frame)
                        
                        # Process based on mode
                        if self.app.pass_through_mode.isChecked():
                            # For pass-through mode, emit the RGB frame
                            pt_start = time.time()
                            
                            # Emit the frame directly from global buffer
                            self.frameProcessed.emit(FRAME_BUFFER_ORIGINAL)  # already RGB
                            
                            # Report pass-through processing time
                            pt_time = (time.time() - pt_start) * 1000  # Convert to ms
                            if frames_captured % 30 == 0:  # Only log occasionally
                                print(f"Pass-through processing time: {pt_time:.3f}ms")
                        elif self.app.rgb_mode.isChecked():
                            # RGB mode
                            processed_array = self.process_frame_array(FRAME_BUFFER_ORIGINAL, 'RGB')
                            if processed_array is not None:
                                # Store result in global output buffer
                                if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != processed_array.shape:
                                    FRAME_BUFFER_OUTPUT = np.empty_like(processed_array)
                                np.copyto(FRAME_BUFFER_OUTPUT, processed_array)
                                # Send NumPy array directly to UI
                                self.frameProcessed.emit(FRAME_BUFFER_OUTPUT)
                        else:
                            # Create grayscale version directly in the global buffer with weighted average
                            if FRAME_BUFFER_GRAYSCALE is None or FRAME_BUFFER_GRAYSCALE.shape != (frame.shape[0], frame.shape[1]):
                                FRAME_BUFFER_GRAYSCALE = np.empty((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            
                            # Calculate grayscale with weighted average
                            FRAME_BUFFER_GRAYSCALE = np.dot(FRAME_BUFFER_ORIGINAL[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                            
                            # Process grayscale
                            processed_array = self.process_frame_array(FRAME_BUFFER_GRAYSCALE, 'L')
                            if processed_array is not None:
                                # Store result in global output buffer
                                if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != processed_array.shape:
                                    FRAME_BUFFER_OUTPUT = np.empty_like(processed_array)
                                np.copyto(FRAME_BUFFER_OUTPUT, processed_array)
                                # Send NumPy array directly to UI
                                self.frameProcessed.emit(FRAME_BUFFER_OUTPUT)
                    else:
                        print(f"Invalid frame format: {frame.shape if frame is not None else None}")
                    
                    # Update timing and adjust capture rate if needed
                    process_end = time.time()
                    process_time = process_end - process_start
                    
                    # Update rolling average of process time
                    self.avg_process_time = 0.8 * self.avg_process_time + 0.2 * process_time
                    
                    # Adaptive interval based on processing time, but prioritize high frame rate
                    # Use a smaller factor (1.05 instead of 1.1) to reduce the safety margin
                    self.min_capture_interval = process_time * 1.05
                    
                    # Allow up to 30fps, but prevent intervals too small to be useful
                    # Removed the 0.2 second minimum which was limiting to 5 FPS
                    self.min_capture_interval = max(0.005, min(0.016, self.min_capture_interval))
                    
                    # Update last capture time
                    self.last_capture_time = time.time()
                    
                    # Report FPS every 5 seconds
                    current_time = time.time()
                    if current_time - last_report_time > 5.0:
                        fps = frames_captured / (current_time - last_report_time)
                        cpu = self.get_cpu_usage()
                        print(f"Camera fps: {fps:.1f}, process time: {process_time*1000:.1f}ms, CPU: {cpu}%")
                        frames_captured = 0
                        last_report_time = current_time
                        
                except Exception as e:
                    print(f"Error capturing/processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    # Exit on device errors, continue for other errors
                    if "device" in str(e).lower() or "resource" in str(e).lower():
                        print("Device error detected, exiting capture loop")
                        break
                
        except Exception as e:
            print(f"Critical error in camera thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.stop_camera()
            print("Camera thread finished")
    
    def process_frame_array(self, array, mode):
        """Apply dithering directly to a NumPy array based on current app settings"""
        try:
            # Get current settings from main app
            alg = self.app.algorithm_combo.currentText()
            thr = self.app.threshold_slider.value()
            contrast_factor = self.app.contrast_slider.value() / 100.0
            pixel_s = self.app.scale_slider.value()
            
            # Access global processing buffer
            global FRAME_BUFFER_PROCESSING
            
            # Fast path for very small images - avoid unnecessary buffer allocation
            is_small = array.size < 100000  # Reduced threshold for better performance
            
            # Optimize array reuse using global buffer
            if is_small:
                # For small images, direct copy is faster than empty_like + copyto
                array_to_dither = array.copy()
            else:
                # For larger images, reuse existing global buffer when possible
                if FRAME_BUFFER_PROCESSING is None or FRAME_BUFFER_PROCESSING.shape != array.shape:
                    FRAME_BUFFER_PROCESSING = np.empty_like(array)
                array_to_dither = FRAME_BUFFER_PROCESSING
                np.copyto(array_to_dither, array)
            
            # Store original shape for ensuring correct upscaling
            orig_shape = array_to_dither.shape
            orig_h, orig_w = orig_shape[:2]
            
            # Apply contrast if needed - only when contrast is significantly different from 1.0
            if abs(contrast_factor - 1.0) > 0.01:
                try:
                    # Apply contrast directly on NumPy array
                    if mode == 'RGB':
                        # Apply to each channel separately
                        for c in range(3):
                            channel = array_to_dither[:,:,c].astype(np.float32)
                            # Simple contrast adjustment formula: f(x) = 128 + contrast_factor * (x - 128)
                            channel = 128 + contrast_factor * (channel - 128)
                            array_to_dither[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
                    else:
                        # Grayscale
                        gray = array_to_dither.astype(np.float32)
                        gray = 128 + contrast_factor * (gray - 128)
                        array_to_dither = np.clip(gray, 0, 255).astype(np.uint8)
                except Exception as e:
                    print(f"Error applying contrast: {e}")
                    # Continue with original array
            
            if pixel_s == 1:
                # Direct processing at original resolution
                if alg == "Floyd-Steinberg":
                    # Apply dithering directly with type optimization
                    result = fs_dither(array_to_dither.astype(np.float32), mode, thr)
                elif alg == "Simple Threshold":
                    # Apply simple threshold directly with minimal allocations
                    if mode == 'RGB':
                        result = simple_threshold_rgb_ps1(array_to_dither, thr)
                    else:
                        result = np.where(array_to_dither < thr, 0, 255).astype(np.uint8)
                else:
                    result = array_to_dither  # Fallback
            else:
                # Pixel scale > 1, we need to handle downscaling and upscaling
                if alg == "Floyd-Steinberg":
                    # The downscale_dither_upscale function already handles the complete pipeline
                    result = downscale_dither_upscale(array_to_dither, thr, pixel_s, mode)
                elif alg == "Simple Threshold":
                    # Perform manual downscale/upscale for simple threshold
                    
                    # First create a downscaled version
                    small_h = max(1, orig_h // pixel_s)
                    small_w = max(1, orig_w // pixel_s)
                    
                    # Downscale using block averaging (simplified)
                    if mode == 'RGB':
                        # RGB block averaging
                        small_arr = np.empty((small_h, small_w, 3), dtype=np.float32)
                        small_arr = block_average_rgb(array_to_dither, small_arr, small_h, small_w, pixel_s)
                        
                        # Apply threshold to downscaled image
                        small_result = simple_threshold_rgb_ps1(small_arr, thr)
                        
                        # Upscale back to original size
                        upscaled = np.empty((orig_h, orig_w, 3), dtype=np.uint8)
                        result = nearest_upscale_rgb(small_result, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
                    else:
                        # Grayscale block averaging
                        small_arr = np.empty((small_h, small_w), dtype=np.float32)
                        small_arr = block_average_gray(array_to_dither, small_arr, small_h, small_w, pixel_s)
                        
                        # Apply threshold to downscaled image
                        small_result = np.where(small_arr < thr, 0, 255).astype(np.uint8)
                        
                        # Upscale back to original size
                        upscaled = np.empty((orig_h, orig_w), dtype=np.uint8)
                        result = nearest_upscale_gray(small_result, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
                else:
                    # Fallback with manual downscale/upscale
                    small_h = max(1, orig_h // pixel_s)
                    small_w = max(1, orig_w // pixel_s)
                    
                    if mode == 'RGB':
                        # RGB downscale
                        small_arr = np.empty((small_h, small_w, 3), dtype=np.float32)
                        small_arr = block_average_rgb(array_to_dither, small_arr, small_h, small_w, pixel_s)
                        # Upscale
                        upscaled = np.empty((orig_h, orig_w, 3), dtype=np.uint8)
                        result = nearest_upscale_rgb(small_arr, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
                    else:
                        # Grayscale downscale
                        small_arr = np.empty((small_h, small_w), dtype=np.float32)
                        small_arr = block_average_gray(array_to_dither, small_arr, small_h, small_w, pixel_s)
                        # Upscale
                        upscaled = np.empty((orig_h, orig_w), dtype=np.uint8)
                        result = nearest_upscale_gray(small_arr, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
            
            # Ensure the result has the same shape as the original
            if mode == 'RGB' and result.shape != orig_shape:
                print(f"Warning: Camera frame result shape {result.shape} doesn't match original {orig_shape}")
                # Attempt to fix the shape
                if len(result.shape) == 2 and len(orig_shape) == 3:
                    # Convert grayscale to RGB
                    rgb_result = np.empty(orig_shape, dtype=np.uint8)
                    for c in range(3):
                        rgb_result[:,:,c] = result
                    result = rgb_result
                    
            # Final dimension check to ensure proper output
            if result.shape != orig_shape and ((mode == 'RGB' and len(orig_shape) == 3) or 
                                              (mode == 'L' and len(orig_shape) == 2)):
                print(f"ERROR: Final result shape {result.shape} still doesn't match original {orig_shape}")
                # Last resort: resize result to match original
                if mode == 'RGB' and len(result.shape) == 3 and len(orig_shape) == 3:
                    # Create a new buffer and manually copy using nearest neighbor
                    fixed_result = np.empty(orig_shape, dtype=np.uint8)
                    result_h, result_w = result.shape[:2]
                    for y in range(orig_h):
                        y_result = min(int(y * result_h / orig_h), result_h-1)
                        for x in range(orig_w):
                            x_result = min(int(x * result_w / orig_w), result_w-1)
                            fixed_result[y, x] = result[y_result, x_result]
                    result = fixed_result
                
            return result
        except Exception as e:
            print(f"Error in process_frame_array: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def stop_camera(self):
        """Clean up camera resources safely"""
        with self.camera_lock:
            if self.camera is not None:
                try:
                    print("Stopping camera device...")
                    # First stop the camera stream
                    self.camera.stop()
                    time.sleep(1.0)  # Wait for camera to stop
                    
                    # Then close the camera completely
                    self.camera.close()
                    time.sleep(0.5)
                    
                    # Set to None to prevent reuse
                    self.camera = None
                    self.camera_initialized = False
                    print("Camera stopped and closed successfully")
                except Exception as e:
                    print(f"Error stopping camera: {e}")
                    import traceback
                    traceback.print_exc()
                    self.camera = None
                    self.camera_initialized = False
            else:
                print("Camera was already None")
        
        # Try to help the system release camera resources - only if CAMERA_SAFE_INIT is enabled
        if config.CAMERA_SAFE_INIT:
            try:
                # Run garbage collection to help release resources
                import gc
                gc.collect()
                time.sleep(0.5)
                
                # Try to force system-level cleanup as well
                import os
                os.system("sudo pkill -f libcamera 2>/dev/null || true")
            except Exception as e:
                print(f"Error during cleanup: {e}")
        else:
            print("Camera device cleanup skipped (disabled in config)")

    def stop(self):
        """Stop the camera capture thread safely"""
        print("CameraCaptureThread stopping...")
        self.is_running = False
        
        # Clean up camera resources
        self.stop_camera()
        
        # For clean thread termination
        self.wait(1000)  # Wait up to 1 second for thread to finish

class DitherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1200, 800)
        
        # Use global buffers instead of instance-specific buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        global FRAME_BUFFER_GRAYSCALE
        global SHARED_ALGORITHM_BUFFER
        global LAST_FRAME_TIME
        global FRAME_COUNT
        
        # Initialize buffer references
        # These are now initialized in main() before app creation
        
        self.showing_original = False # Show dithered version first by default
        
        # Initialize camera mode as disabled by default
        self.camera_mode_active = False
        self.capture_thread = None
        
        # Flag for frame capture
        self.capture_frame_requested = False
        
        # Freeze frame variables
        self.freeze_frame = None
        self.freeze_frame_time = 0
        self.freeze_frame_duration = 2.0  # Freeze for 2 seconds
        
        self.setup_ui()

    def stop_camera(self):
        """Stop camera capture, ensuring proper synchronization"""
        print("Stopping camera...")
        self.camera_mode_active = False  # Set this first to prevent new frames processing
        
        # Stop thread with shorter timeout
        thread_stop_timeout = 2.0  # Reduced from 3.0 seconds
        stop_successful = True
        
        # Stop capture thread
        if self.capture_thread and self.capture_thread.isRunning():
            print("Stopping camera thread...")
            self.capture_thread.stop()
            
            # Wait with timeout for thread to finish
            start_time = time.time()
            while self.capture_thread.isRunning() and (time.time() - start_time) < thread_stop_timeout:
                time.sleep(0.05)  # Check more frequently
            
            if self.capture_thread.isRunning():
                print("Warning: Camera thread did not stop within timeout")
                stop_successful = False
            else:
                print("Camera thread stopped successfully")
        
        # Run additional system-level cleanup - only if CAMERA_SAFE_INIT is enabled
        if config.CAMERA_SAFE_INIT:
            try:
                # Force close any existing camera processes at system level
                import os
                os.system("sudo pkill -f libcamera")
                time.sleep(1.0)  # Reduced wait time
                
                # Run garbage collection for Python objects
                import gc
                gc.collect()
            except Exception as e:
                print(f"System-level cleanup failed (non-critical): {e}")
        else:
            print("Camera safe cleanup skipped (disabled in config)")
        
        # Explicitly delete thread object to ensure it's fully released
        if self.capture_thread:
            try:
                del self.capture_thread
                self.capture_thread = None
            except:
                pass
        
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
        
        # Give the system a moment to fully release camera resources
        time.sleep(1.0)  # Reduced wait time

    def start_camera(self):
        """Start camera capture and direct processing, ensuring proper synchronization"""
        if not PICAMERA_AVAILABLE:
            print("Camera not available")
            return
            
        # If camera is already active, stop it first to ensure clean state
        if self.camera_mode_active:
            print("Camera already active, stopping first")
            self.stop_camera()
            # Wait longer to ensure everything is stopped
            time.sleep(2.0)
        
        print("Starting camera...")
        
        # Additional system cleanup for camera - only if CAMERA_SAFE_INIT is enabled
        if config.CAMERA_SAFE_INIT:
            try:
                # First try to clean up any existing camera instances at the system level
                import subprocess
                import os
                
                # Run cleanup commands
                os.system("sudo pkill -f libcamera")
                time.sleep(1.0)  # Reduced wait time
                
                # Try to reset the camera system
                try:
                    print("Attempting system-level camera reset...")
                    subprocess.run(["sudo", "libcamera-still", "-t", "1", "--immediate"], 
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=2)
                except:
                    print("System tool timeout, continuing anyway")
                    
                time.sleep(1.0)  # Reduced wait time
                
                # Run garbage collection
                import gc
                gc.collect()
            except Exception as e:
                print(f"System-level cleanup failed (non-critical): {e}")
        else:
            print("Camera safe initialization skipped (disabled in config)")
            
        # Create fresh thread instance that processes frames directly with higher priority
        try:
            self.capture_thread = CameraCaptureThread(self)
            self.capture_thread.frameProcessed.connect(self.update_camera_frame)
            
            # Set higher thread priority for better performance
            self.capture_thread.setPriority(QThread.Priority.HighPriority)
            
            # Start camera thread
            print("Starting camera thread...")
            
            if not self.capture_thread.isRunning():
                self.capture_thread.start()
                print("Camera thread started")
                time.sleep(1.0)  # Reduced wait time
            
            # Check if thread actually started
            if not self.capture_thread.isRunning():
                print("Failed to start camera thread")
                self.stop_camera()
                return
            
            # Update UI
            self.camera_button.setText("Stop Camera")
            self.capture_button.setEnabled(True)
            self.open_button.setEnabled(False)  # Disable file open during camera mode
            self.camera_mode_active = True
            
            # Clear current image
            self.showing_original = False
            FRAME_BUFFER_ORIGINAL = None
            FRAME_BUFFER_OUTPUT = None
            self.toggle_button.setEnabled(False)
            
            print("Camera mode started successfully")
        except Exception as e:
            print(f"Error starting camera: {e}")
            # Try to clean up on error
            self.stop_camera()

    def toggle_camera_mode(self):
        if self.camera_mode_active:
            self.stop_camera()
        else:
            self.start_camera()

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
        
        # Add pass-through mode checkbox
        self.pass_through_mode = QCheckBox("Pass-through")
        self.pass_through_mode.setChecked(False)
        self.pass_through_mode.stateChanged.connect(self.pass_through_changed)
        checkbox_layout.addWidget(self.pass_through_mode)
        
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

    def handle_zoom_slider_change(self, value):
        """Handle zoom slider changes by updating the image viewer zoom"""
        zoom_factor = value / 100.0  # Convert percentage to factor
        self.zoom_label.setText(f"Zoom: {value}%")
        
        # Update image viewer zoom (without triggering feedback loop)
        self.image_viewer.set_zoom(zoom_factor)
    
    def update_controls_from_zoom_factor(self, zoom_factor):
        """Update zoom controls when zoom changes from image viewer"""
        zoom_percent = int(zoom_factor * 100)
        
        # Update slider without triggering valueChanged signal again
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_percent)
        self.zoom_slider.blockSignals(False)
        
        # Update label
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")
        
    def toggle_image_display(self):
        """Toggle between showing original and dithered image"""
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_OUTPUT
        
        if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_OUTPUT is None:
            return  # No images to toggle between
            
        self.showing_original = not self.showing_original
        
        if self.showing_original:
            self.toggle_button.setText("Switch to Dithered Image")
            self.image_viewer.set_image(FRAME_BUFFER_ORIGINAL)
        else:
            self.toggle_button.setText("Switch to Original Image")
            self.image_viewer.set_image(FRAME_BUFFER_OUTPUT)
            
    def update_camera_frame(self, frame_array):
        """Update the image display with camera frame"""
        if not self.camera_mode_active:
            return  # Ignore frames if camera is no longer active
        
        # Check if we're in freeze frame mode
        current_time = time.time()
        if self.freeze_frame is not None:
            # If we're still within the freeze period, continue showing the frozen frame
            if current_time - self.freeze_frame_time < self.freeze_frame_duration:
                return  # Keep displaying the frozen frame
            else:
                # Freeze period is over, clear the freeze frame
                self.freeze_frame = None
                print("Freeze frame period ended, returning to live view")
            
        # Set the frame to the image viewer
        self.image_viewer.set_image(frame_array)
        
        # Enable toggle button if we have both original and output frames
        global FRAME_BUFFER_ORIGINAL, FRAME_BUFFER_OUTPUT
        if FRAME_BUFFER_ORIGINAL is not None and FRAME_BUFFER_OUTPUT is not None:
            self.toggle_button.setEnabled(True)
            
        # Check if capture was requested
        if self.capture_frame_requested:
            self.capture_frame_requested = False
            
            # Store the current frame as the freeze frame
            if self.showing_original and FRAME_BUFFER_ORIGINAL is not None:
                self.freeze_frame = FRAME_BUFFER_ORIGINAL.copy()
            elif FRAME_BUFFER_OUTPUT is not None:
                self.freeze_frame = FRAME_BUFFER_OUTPUT.copy()
            else:
                self.freeze_frame = frame_array.copy()
                
            # Save the current captured frame as FRAME_BUFFER_ORIGINAL
            global FRAME_BUFFER_ORIGINAL
            if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_ORIGINAL.shape != frame_array.shape:
                FRAME_BUFFER_ORIGINAL = np.empty_like(frame_array)
            np.copyto(FRAME_BUFFER_ORIGINAL, frame_array)
            
            # Save FRAME_BUFFER_ORIGINAL
            self.save_captured_image()
            
            # Display the frozen frame
            self.image_viewer.set_image(self.freeze_frame)
            self.freeze_frame_time = current_time
            print(f"Captured frame frozen for {self.freeze_frame_duration} seconds")
            
    def capture_frame(self):
        """Flag to capture and save the next frame"""
        self.capture_frame_requested = True
        
    def algorithm_changed(self, index):
        """Handle algorithm change and reapply if auto-render is checked"""
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def threshold_changed(self, value):
        """Update threshold label and reapply if auto-render is checked"""
        self.threshold_label.setText(f"Threshold: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def contrast_changed(self, value):
        """Update contrast label and reapply if auto-render is checked"""
        contrast = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast:.1f}")
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def scale_changed(self, value):
        """Update scale label and reapply if auto-render is checked"""
        self.scale_label.setText(f"Pixel Scale: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def rgb_changed(self, state):
        """Handle toggling between RGB and grayscale mode"""
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def pass_through_changed(self, state):
        """Handle toggling pass-through mode"""
        algorithm_enabled = not self.pass_through_mode.isChecked()
        
        # Enable or disable algorithm controls
        self.algorithm_combo.setEnabled(algorithm_enabled)
        self.threshold_slider.setEnabled(algorithm_enabled)
        self.rgb_mode.setEnabled(algorithm_enabled)
        
        if self.auto_render.isChecked():
            self.apply_dither()
            
    def open_image(self):
        """Open an image file through file dialog"""
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", 
            options=options
        )
        
        if file_path:
            try:
                # Use OpenCV to read image if available
                try:
                    import cv2
                    img = cv2.imread(file_path)
                    if img is None:
                        raise ImportError("Failed to read with OpenCV")
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except ImportError:
                    # Fall back to PIL
                    from PIL import Image
                    img = np.array(Image.open(file_path).convert("RGB"))
                
                # Update global buffer
                global FRAME_BUFFER_ORIGINAL
                FRAME_BUFFER_ORIGINAL = img
                
                # Display the loaded image
                self.image_viewer.set_image(img)
                self.showing_original = True
                self.toggle_button.setText("Switch to Dithered Image")
                self.toggle_button.setEnabled(False)  # Disable until dithered version exists
                
                # Enable buttons
                self.save_button.setEnabled(True)
                self.apply_button.setEnabled(True)
                
                # Apply dither if auto-render is checked
                if self.auto_render.isChecked():
                    self.apply_dither()
            except Exception as e:
                print(f"Error opening image: {e}")
                import traceback
                traceback.print_exc()
                
    def apply_dither(self):
        """Apply dithering to the current image"""
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_OUTPUT
        
        if FRAME_BUFFER_ORIGINAL is None:
            return  # No image to process
            
        try:
            # Get current settings
            alg = self.algorithm_combo.currentText()
            thr = self.threshold_slider.value()
            contrast_factor = self.contrast_slider.value() / 100.0
            pixel_s = self.scale_slider.value()
            
            # Check if we're in pass-through mode
            if self.pass_through_mode.isChecked():
                # Just copy the original to output
                if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != FRAME_BUFFER_ORIGINAL.shape:
                    FRAME_BUFFER_OUTPUT = np.empty_like(FRAME_BUFFER_ORIGINAL)
                np.copyto(FRAME_BUFFER_OUTPUT, FRAME_BUFFER_ORIGINAL)
            else:
                # Create a copy of the original for processing
                work_copy = FRAME_BUFFER_ORIGINAL.copy()
                orig_shape = work_copy.shape  # Save original shape for upscaling later
                orig_h, orig_w = orig_shape[:2]
                
                # Apply contrast adjustment if needed
                if abs(contrast_factor - 1.0) > 0.01:
                    work_copy = work_copy.astype(np.float32)
                    work_copy = 128 + contrast_factor * (work_copy - 128)
                    work_copy = np.clip(work_copy, 0, 255).astype(np.uint8)
                
                # Apply dithering based on mode and pixel scale
                if pixel_s == 1:
                    # Process at original resolution
                    if self.rgb_mode.isChecked():
                        # Process for RGB mode
                        if alg == "Floyd-Steinberg":
                            result = fs_dither(work_copy.astype(np.float32), 'RGB', thr)
                        else:  # Simple Threshold
                            result = simple_threshold_rgb_ps1(work_copy, thr)
                    else:
                        # Process for Grayscale
                        # Convert to grayscale first
                        gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                        
                        if alg == "Floyd-Steinberg":
                            result = fs_dither(gray.astype(np.float32), 'L', thr)
                        else:  # Simple Threshold
                            result = np.where(gray < thr, 0, 255).astype(np.uint8)
                else:
                    # Process with downscaling and upscaling for pixel_s > 1
                    if self.rgb_mode.isChecked():
                        # RGB mode
                        if alg == "Floyd-Steinberg":
                            # This function already handles the complete pipeline
                            result = downscale_dither_upscale(work_copy, thr, pixel_s, 'RGB')
                        else:  # Simple Threshold
                            # Manual downscale and upscale
                            # First determine size of downscaled image
                            small_h = max(1, orig_h // pixel_s)
                            small_w = max(1, orig_w // pixel_s)
                            
                            # Perform block averaging for downscaling
                            small_arr = np.empty((small_h, small_w, 3), dtype=np.float32)
                            small_arr = block_average_rgb(work_copy, small_arr, small_h, small_w, pixel_s)
                            
                            # Apply threshold to downscaled image
                            small_result = simple_threshold_rgb_ps1(small_arr, thr)
                            
                            # Upscale back to original size
                            upscaled = np.empty((orig_h, orig_w, 3), dtype=np.uint8)
                            result = nearest_upscale_rgb(small_result, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
                    else:
                        # Grayscale mode
                        # Convert to grayscale first
                        gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                        orig_h, orig_w = gray.shape
                        
                        if alg == "Floyd-Steinberg":
                            # This function already handles the complete pipeline
                            result = downscale_dither_upscale(gray, thr, pixel_s, 'L')
                        else:  # Simple Threshold
                            # Manual downscale and upscale
                            # First determine size of downscaled image
                            small_h = max(1, orig_h // pixel_s)
                            small_w = max(1, orig_w // pixel_s)
                            
                            # Perform block averaging for downscaling
                            small_arr = np.empty((small_h, small_w), dtype=np.float32)
                            small_arr = block_average_gray(gray, small_arr, small_h, small_w, pixel_s)
                            
                            # Apply threshold to downscaled image
                            small_result = np.where(small_arr < thr, 0, 255).astype(np.uint8)
                            
                            # Upscale back to original size
                            upscaled = np.empty((orig_h, orig_w), dtype=np.uint8)
                            result = nearest_upscale_gray(small_result, upscaled, orig_h, orig_w, small_h, small_w, pixel_s)
                
                # Ensure the result has the same shape as the original
                if self.rgb_mode.isChecked() and result.shape != orig_shape:
                    print(f"Warning: Result shape {result.shape} doesn't match original {orig_shape}")
                    # Attempt to fix the shape
                    if len(result.shape) == 2 and len(orig_shape) == 3:
                        # Convert grayscale to RGB
                        rgb_result = np.empty(orig_shape, dtype=np.uint8)
                        for c in range(3):
                            rgb_result[:,:,c] = result
                        result = rgb_result
                
                # Update output buffer
                FRAME_BUFFER_OUTPUT = result
            
            # Update display if not showing original
            if not self.showing_original:
                self.image_viewer.set_image(FRAME_BUFFER_OUTPUT)
            
            # Enable toggle button now that we have both images
            self.toggle_button.setEnabled(True)
            
            # Enable save button
            self.save_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error applying dither: {e}")
            import traceback
            traceback.print_exc()
            
    def save_image(self):
        """Save the current image (original or dithered)"""
        if self.showing_original and FRAME_BUFFER_ORIGINAL is None:
            return
        if not self.showing_original and FRAME_BUFFER_OUTPUT is None:
            return
            
        # Determine which image to save
        img_to_save = FRAME_BUFFER_ORIGINAL if self.showing_original else FRAME_BUFFER_OUTPUT
        self._save_image_to_file(img_to_save)
        
    def save_captured_image(self):
        """Save the captured frame from the camera"""
        if FRAME_BUFFER_ORIGINAL is None:
            print("No frame to save")
            return
            
        self._save_image_to_file(FRAME_BUFFER_ORIGINAL)
            
    def _save_image_to_file(self, img_to_save):
        """Common method to save an image to file with proper extension handling"""
        options = QFileDialog.Option.ReadOnly
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", 
            options=options
        )
        
        if not file_path:
            return
            
        # Ensure file has the correct extension based on selected filter
        if selected_filter == "PNG Files (*.png)" and not file_path.lower().endswith('.png'):
            file_path += '.png'
        elif selected_filter == "JPEG Files (*.jpg)" and not file_path.lower().endswith(('.jpg', '.jpeg')):
            file_path += '.jpg'
            
        try:
            # Use PIL to save the image (more reliable than OpenCV for this case)
            try:
                from PIL import Image
                if len(img_to_save.shape) == 2:  # Grayscale
                    Image.fromarray(img_to_save).save(file_path)
                else:  # RGB
                    Image.fromarray(img_to_save).save(file_path)
                print(f"Image saved to {file_path}")
            except ImportError:
                # Fall back to OpenCV if PIL is not available
                import cv2
                # Convert from RGB to BGR for OpenCV
                img_to_save_bgr = img_to_save.copy()
                if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
                    img_to_save_bgr = img_to_save_bgr[:, :, ::-1]
                success = cv2.imwrite(file_path, img_to_save_bgr)
                if not success:
                    raise RuntimeError(f"OpenCV failed to save image to {file_path}")
                print(f"Image saved to {file_path} using OpenCV")
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main application entry point with initialization of global frame buffers"""
    # Make sure time module is accessible throughout
    import time as time_module
    
    # Initialize global frame handling variables
    global FRAME_BUFFER_ORIGINAL
    global FRAME_BUFFER_PROCESSING
    global FRAME_BUFFER_OUTPUT
    global FRAME_BUFFER_GRAYSCALE
    global SHARED_ALGORITHM_BUFFER
    global LAST_FRAME_TIME
    global FRAME_COUNT
    
    # Initialize buffers with default size (640x480) to avoid first-time allocation lag
    default_width = 640
    default_height = 480
    
    # Initialize RGB buffers (3 channels)
    FRAME_BUFFER_ORIGINAL = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    FRAME_BUFFER_PROCESSING = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    FRAME_BUFFER_OUTPUT = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    
    # Initialize grayscale buffer
    FRAME_BUFFER_GRAYSCALE = np.zeros((default_height, default_width), dtype=np.uint8)
    
    # Initialize algorithm buffers for both RGB and grayscale
    SHARED_ALGORITHM_BUFFER = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    
    LAST_FRAME_TIME = time_module.time()
    FRAME_COUNT = 0
    
    # Only run camera cleanup if CAMERA_SAFE_INIT is enabled in config
    if config.CAMERA_SAFE_INIT:
        try:
            import os
            import subprocess
            
            print("Performing thorough camera system cleanup at startup...")
            
            # First kill any existing camera processes
            os.system("sudo pkill -f libcamera")
            time_module.sleep(2.0)
            
            # Also kill any Python processes that might be using the camera
            os.system("sudo pkill -f python.*picamera")
            time_module.sleep(1.0)
            
            # Run a quick camera capture to reset the system
            try:
                subprocess.run(["sudo", "libcamera-still", "-t", "1", "--immediate"], 
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=3)
            except:
                print("libcamera-still command timed out, this is normal")
                
            # Give system time to release resources
            time_module.sleep(2.0)
            
            # Force the libcamera system service to restart
            os.system("sudo systemctl restart libcamera.service 2>/dev/null || true")
            time_module.sleep(2.0)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("Camera system reset complete")
            
        except Exception as e:
            print(f"Camera system cleanup at startup failed (non-critical): {e}")
    else:
        print("Camera safe initialization skipped (disabled in config)")

    app = QApplication(sys.argv)
    window = DitherApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
