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
                    self.min_capture_interval = max(0.01, min(0.033, self.min_capture_interval))
                    
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
            
            # Fast path selection based on algorithm
            if alg == "Floyd-Steinberg":
                if pixel_s == 1:
                    # Apply dithering directly with type optimization
                    result = fs_dither(array_to_dither.astype(np.float32), mode, thr)
                else:
                    # Use the optimized downscale-dither-upscale pipeline
                    result = downscale_dither_upscale(array_to_dither, thr, pixel_s, mode)
            elif alg == "Simple Threshold":
                if pixel_s == 1:
                    # Apply simple threshold directly with minimal allocations
                    if mode == 'RGB':
                        result = simple_threshold_rgb_ps1(array_to_dither, thr)
                    else:
                        result = np.where(array_to_dither < thr, 0, 255).astype(np.uint8)
                else:
                    # Use block-based approach
                    orig_h, orig_w = array_to_dither.shape[:2]
                    result = simple_threshold_dither(array_to_dither, mode, pixel_s, orig_w, orig_h, thr)
            else:
                result = array_to_dither  # Fallback
                
            return result
        except Exception as e:
            print(f"Error in process_frame_array: {e}")
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
