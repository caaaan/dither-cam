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
        
        # Try to help the system release camera resources
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

    def stop(self):
        """Stop the thread safely"""
        print("Requesting camera thread to stop...")
        self.is_running = False
        # Give thread a moment to notice the flag
        time.sleep(0.2)
        self.stop_camera()  # Stop the camera right away

class FrameProcessingThread(QThread):
    frameProcessed = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue, app_instance):
        super().__init__()
        self.is_running = False  # Start as not running
        self.frame_queue = frame_queue
        self.app = app_instance  # Reference to the main app for dithering settings
        print("Processing thread initialized")
        
        # Access global buffers instead of creating thread-local buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        
        # Thread synchronization - detect if we're falling behind
        self.processed_count = 0
        self.queue_full_count = 0
        self.last_process_time = 0
        self.avg_process_time = 0.02  # Initial guess: 20ms per frame (optimistic for 30 FPS)

    def run(self):
        print("Frame processing thread starting...")
        self.is_running = True
        
        # Access global buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        global FRAME_BUFFER_GRAYSCALE
        global LAST_FRAME_TIME
        global FRAME_COUNT
        global LAST_FPS
        global FRAME_PROCESS_TIME
        
        # Performance tracking
        frames_processed = 0
        last_report_time = time.time()
        
        while self.is_running:
            try:
                # Use timeout to ensure we regularly check if thread should stop
                try:
                    # Non-blocking queue get with timeout for better responsiveness
                    # Reduced timeout for higher framerates
                    frame = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    # No frames to process, check queue size for monitoring
                    if self.frame_queue.qsize() > self.frame_queue.maxsize * 0.8:
                        self.queue_full_count += 1
                        if self.queue_full_count % 5 == 0:
                            print(f"Warning: Processing thread falling behind, queue {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
                    continue
                    
                process_start = time.time()
                frames_processed += 1
                
                # Reset queue full counter since we got a frame
                self.queue_full_count = 0
                
                # Use direct NumPy array operations for BGR to RGB conversion
                frame = frame[:, :, ::-1].copy()
                
                # Make sure we have a proper RGB frame
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    try:
                        # Store in global original buffer
                        if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_ORIGINAL.shape != frame.shape:
                            FRAME_BUFFER_ORIGINAL = np.empty_like(frame)
                        np.copyto(FRAME_BUFFER_ORIGINAL, frame)
                        
                        # Fast path for pass-through mode
                        if self.app.pass_through_mode.isChecked():
                            # In pass-through mode, just emit the RGB frame directly from global buffer
                            if self.is_running:
                                self.frameProcessed.emit(FRAME_BUFFER_ORIGINAL)
                        else:
                            # Process with minimum memory allocations
                            if not self.app.rgb_mode.isChecked():
                                # Create grayscale version in global grayscale buffer
                                if FRAME_BUFFER_GRAYSCALE is None or FRAME_BUFFER_GRAYSCALE.shape != (frame.shape[0], frame.shape[1]):
                                    FRAME_BUFFER_GRAYSCALE = np.empty((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                                
                                # Calculate grayscale with weighted average
                                FRAME_BUFFER_GRAYSCALE = np.dot(FRAME_BUFFER_ORIGINAL[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                                
                                # Process grayscale NumPy array directly
                                processed_array = self.process_frame_array(FRAME_BUFFER_GRAYSCALE, 'L')
                            else:
                                # Process RGB NumPy array directly
                                processed_array = self.process_frame_array(FRAME_BUFFER_ORIGINAL, 'RGB')
                            
                            if processed_array is not None and self.is_running:
                                # Store result in global output buffer
                                if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != processed_array.shape:
                                    FRAME_BUFFER_OUTPUT = np.empty_like(processed_array)
                                np.copyto(FRAME_BUFFER_OUTPUT, processed_array)
                                self.frameProcessed.emit(FRAME_BUFFER_OUTPUT)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                else:
                    print(f"Invalid frame format: {frame.shape if frame is not None else None}")
            except Exception as e:
                print(f"Error in processing thread main loop: {e}")
                # Don't exit the loop for occasional errors
            
            # Update processing time statistics for adaptive timing
            process_end = time.time()
            process_time = process_end - process_start
            
            # Exponential moving average for process time with higher weight on recent times
            self.avg_process_time = 0.7 * self.avg_process_time + 0.3 * process_time
            
            # Report FPS every 3 seconds (more frequent reporting)
            self.processed_count += 1
            current_time = time.time()
            if current_time - last_report_time > 3.0:
                fps = self.processed_count / (current_time - last_report_time)
                print(f"Frame processing rate: {fps:.1f} FPS, avg process time: {self.avg_process_time*1000:.1f}ms")
                self.processed_count = 0
                last_report_time = current_time
        
        print("Processing thread stopped")

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

        # Single QImage buffer for display
        self.qimage_buffer = None
        self.current_format = None
        self.current_size = None

    def update_frame(self, frame_array):
        """Efficiently update the display with a NumPy array (RGB or grayscale)."""
        import numpy as np
        from PyQt6.QtGui import QImage, QPixmap
        
        # Determine format - assumes frame_array is already in RGB order (not BGR)
        if frame_array.ndim == 3 and frame_array.shape[2] == 3:
            # QImage.Format_RGB888 expects data in RGB order
            fmt = QImage.Format.Format_RGB888
            bytes_per_line = frame_array.shape[1] * 3
        elif frame_array.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8
            bytes_per_line = frame_array.shape[1]
        else:
            raise ValueError(f"Unsupported frame shape for update_frame: {frame_array.shape}")
        
        size = (frame_array.shape[1], frame_array.shape[0])
        
        # Only recreate QImage if size or format changes - optimized buffer reuse
        if self.qimage_buffer is None or self.current_format != fmt or self.current_size != size:
            # Create a new QImage from the NumPy array
            self.qimage_buffer = QImage(frame_array.data, size[0], size[1], bytes_per_line, fmt)
            self.current_format = fmt
            self.current_size = size
            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(self.qimage_buffer)
        else:
            # Update existing QImage buffer with new data without allocating new memory
            self.qimage_buffer = QImage(frame_array.data, size[0], size[1], bytes_per_line, fmt)
            # Create QPixmap from updated QImage
            pixmap = QPixmap.fromImage(self.qimage_buffer)
        
        # Update the image in the label with the newly created pixmap
        self.set_image(pixmap)

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
        
        # Use global buffers instead of instance-specific buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        global FRAME_BUFFER_GRAYSCALE
        global SHARED_ALGORITHM_BUFFER
        global LAST_FRAME_TIME
        global FRAME_COUNT
        
        # Set initial values
        FRAME_BUFFER_ORIGINAL = None
        FRAME_BUFFER_PROCESSING = None
        FRAME_BUFFER_OUTPUT = None
        FRAME_BUFFER_GRAYSCALE = None
        SHARED_ALGORITHM_BUFFER = None
        LAST_FRAME_TIME = time.time()
        FRAME_COUNT = 0
        
        self.showing_original = False # Show dithered version first by default
        
        # Initialize camera mode as disabled by default
        self.camera_mode_active = False
        self.capture_thread = None
        
        # Flag for frame capture
        self.capture_frame_requested = False
        
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
    
    def toggle_camera_mode(self):
        if self.camera_mode_active:
            self.stop_camera()
        else:
            self.start_camera()
    
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
        
        # Additional system cleanup for camera - reduced wait times
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
        
        # Run additional system-level cleanup - reduced wait times
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
    
    def capture_frame(self):
        """Trigger the camera to capture a single frame and save it"""
        if not self.camera_mode_active:
            print("Camera not active, cannot capture frame")
            return
        
        if not self.capture_thread or not self.capture_thread.isRunning():
            print("Camera thread not running")
            return
        
        # Flag that we want to keep the next processed frame
        self.capture_frame_requested = True
        print("Frame capture requested - waiting for next frame...")
        
        # The update_camera_frame method will handle saving the next processed frame
    
    def update_camera_frame(self, dithered_img):
        """Update the UI with the processed camera frame"""
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_OUTPUT
        
        if not self.camera_mode_active:
            print("Camera not active, frame update ignored")
            return
        
        import numpy as np
        # Make sure we have a NumPy array
        if not isinstance(dithered_img, np.ndarray):
            print(f"update_camera_frame: Unsupported image type {type(dithered_img)}")
            self.image_viewer.set_image(None)
            return
        
        # Display the dithered frame
        if dithered_img is not None:
            # Check if we need to capture this frame
            if self.capture_frame_requested:
                print("Captured frame received")
                self.capture_frame_requested = False  # Reset flag
                
                # Store in global original buffer
                if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_ORIGINAL.shape != dithered_img.shape:
                    FRAME_BUFFER_ORIGINAL = np.empty_like(dithered_img)
                np.copyto(FRAME_BUFFER_ORIGINAL, dithered_img)
                
                print(f"Frame captured: array shape={dithered_img.shape}")
                print("Stopping camera after frame capture...")
                self.stop_camera()
                try:
                    print("Performing additional system-level cleanup...")
                    import os
                    import time
                    os.system("sudo pkill -f libcamera")
                    time.sleep(2.0)
                    import gc
                    gc.collect()
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Additional cleanup error (non-critical): {e}")
                print("Applying dithering to captured frame...")
                self.apply_dither()
                if FRAME_BUFFER_OUTPUT is not None:
                    self.save_button.setEnabled(True)
                    self.toggle_button.setEnabled(True)
                    print("Dithering applied successfully")
                else:
                    print("Failed to apply dithering to captured frame")
                return
            
            # For normal camera display, update the display
            self.image_viewer.update_frame(dithered_img)
            
            # Store current output in global buffer
            if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != dithered_img.shape:
                FRAME_BUFFER_OUTPUT = np.empty_like(dithered_img)
            np.copyto(FRAME_BUFFER_OUTPUT, dithered_img)
            
            self.save_button.setEnabled(True)
        else:
            print("Warning: Received None image in update_camera_frame")
    
    def toggle_image_display(self):
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_OUTPUT
        
        if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_OUTPUT is None:
            return
        
        # Update showing_original flag
        self.showing_original = not self.showing_original
        
        # If we're in pass-through mode, disable it when switching to dithered view
        if not self.showing_original and self.pass_through_mode.isChecked():
            self.pass_through_mode.setChecked(False)
        
        if self.showing_original:
            self.display_image(FRAME_BUFFER_ORIGINAL)
            self.toggle_button.setText("Switch to Dithered Image")
        else:
            self.display_image(FRAME_BUFFER_OUTPUT)
            self.toggle_button.setText("Switch to Original Image")
    
    def open_image(self):
        """Open an image file and convert to NumPy array"""
        global FRAME_BUFFER_ORIGINAL
        
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
            # Load the image with OpenCV - simple and direct
            import cv2
            img_array = cv2.imread(path)
            if img_array is None:
                raise ValueError(f"Failed to load image from {path}")
                
            # Convert BGR to RGB using NumPy slicing and store in global buffer
            img_array_rgb = img_array[:, :, ::-1]  # BGR to RGB conversion
            
            # Store in global original buffer
            if FRAME_BUFFER_ORIGINAL is None or FRAME_BUFFER_ORIGINAL.shape != img_array_rgb.shape:
                FRAME_BUFFER_ORIGINAL = np.empty_like(img_array_rgb)
            np.copyto(FRAME_BUFFER_ORIGINAL, img_array_rgb)
            
            print(f"Loaded image from {path}, shape={FRAME_BUFFER_ORIGINAL.shape}")
            
            # Convert to grayscale if needed based on current mode
            if not self.rgb_mode.isChecked() and len(FRAME_BUFFER_ORIGINAL.shape) == 3:
                # Convert to grayscale using NumPy weighted average
                gray = np.dot(FRAME_BUFFER_ORIGINAL[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                FRAME_BUFFER_ORIGINAL = gray
            
            # Apply dithering directly to array
            self.apply_dither()
            self.image_viewer.set_zoom_level(1.0) # Reset zoom for the new image view

            if FRAME_BUFFER_OUTPUT is not None:
                # Dithering was successful
                self.toggle_button.setText("Switch to Original Image")
                self.save_button.setEnabled(True)
                self.toggle_button.setEnabled(True)
                self.showing_original = False
            else:
                # Initial dithering failed, fall back to showing original
                self.showing_original = True
                self.display_image(FRAME_BUFFER_ORIGINAL)
                self.toggle_button.setText("Switch to Dithered Image")
                self.save_button.setEnabled(False)
                self.toggle_button.setEnabled(False)
                
        except Exception as e:
            print(f"Error opening image: {e}")
            import traceback
            traceback.print_exc()
            FRAME_BUFFER_ORIGINAL = None
            FRAME_BUFFER_OUTPUT = None
            self.image_viewer.set_image(None) 
            self.save_button.setEnabled(False)
            self.toggle_button.setEnabled(False)
            self.showing_original = False
    
    def display_image(self, image, is_bgr_data=False):
        """Display an image in the image viewer, ensuring correct RGB color order"""
        import numpy as np
        
        if image is None:
            self.image_viewer.set_image(None)
            return
        
        # Make sure we're dealing with a NumPy array
        if not isinstance(image, np.ndarray):
            print(f"display_image: Image is not a NumPy array, type={type(image)}")
            self.image_viewer.set_image(None)
            return
        
        # Convert BGR to RGB if needed using NumPy (avoid OpenCV dependency)
        if is_bgr_data and len(image.shape) == 3 and image.shape[2] == 3:
            # Simply swap the channels using NumPy slice operations
            image = image[:, :, ::-1].copy()
        
        try:
            # Update the image viewer with the RGB NumPy array
            self.image_viewer.update_frame(image)
        except Exception as e:
            print(f"Error updating image viewer: {e}")
            import traceback
            traceback.print_exc()
            self.image_viewer.set_image(None)
    
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
        # Pre-allocate shared buffer if needed based on algorithm
        global SHARED_ALGORITHM_BUFFER, FRAME_BUFFER_PROCESSING
        
        # If we have a processing buffer, ensure shared buffer is ready
        if FRAME_BUFFER_PROCESSING is not None:
            alg = self.algorithm_combo.currentText()
            
            # Check if we're in RGB or grayscale mode to create the correct shape
            is_rgb = self.rgb_mode.isChecked()
            
            try:
                if alg == "Floyd-Steinberg":
                    # Float32 for Floyd-Steinberg
                    SHARED_ALGORITHM_BUFFER = np.empty_like(FRAME_BUFFER_PROCESSING, dtype=np.float32)
                    print(f"Pre-allocated float32 buffer for Floyd-Steinberg, shape={SHARED_ALGORITHM_BUFFER.shape}")
                else:
                    # Uint8 for Simple Threshold
                    SHARED_ALGORITHM_BUFFER = np.empty_like(FRAME_BUFFER_PROCESSING, dtype=np.uint8)
                    print(f"Pre-allocated uint8 buffer for Simple Threshold, shape={SHARED_ALGORITHM_BUFFER.shape}")
            except Exception as e:
                print(f"Warning: Failed to pre-allocate buffer: {e}, will create on demand")
                SHARED_ALGORITHM_BUFFER = None
        
        # Continue with render if auto-render is on
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def rgb_changed(self):
        """Called when the RGB checkbox state changes"""
        print(f"RGB mode changed to: {self.rgb_mode.isChecked()}")
        
        # Reset the shared algorithm buffer when switching modes to force recreation with correct shape
        global SHARED_ALGORITHM_BUFFER
        SHARED_ALGORITHM_BUFFER = None
        
        # Also reset the processing buffer to ensure the shape is correct
        global FRAME_BUFFER_PROCESSING
        FRAME_BUFFER_PROCESSING = None
        
        # Update camera processing immediately if in camera mode
        if self.camera_mode_active:
            print("Applying RGB mode change to camera feed")
            
        # Apply dithering to static images if auto-render is on
        if self.auto_render.isChecked() and FRAME_BUFFER_ORIGINAL is not None:
            self.apply_dither()
    
    def apply_dither(self):
        """Apply selected dithering algorithm to the original image"""
        # Use global buffers
        global FRAME_BUFFER_ORIGINAL
        global FRAME_BUFFER_PROCESSING
        global FRAME_BUFFER_OUTPUT
        global FRAME_BUFFER_GRAYSCALE
        
        # Original array is stored in the global buffer
        if FRAME_BUFFER_ORIGINAL is None:
            return
        
        # In pass-through mode, just display the original image
        if self.pass_through_mode.isChecked():
            # Copy original to output
            if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != FRAME_BUFFER_ORIGINAL.shape:
                FRAME_BUFFER_OUTPUT = np.empty_like(FRAME_BUFFER_ORIGINAL)
            np.copyto(FRAME_BUFFER_OUTPUT, FRAME_BUFFER_ORIGINAL)
            
            self.showing_original = True
            self.toggle_button.setText("Switch to Dithered Image")
            self.toggle_button.setEnabled(True)
            self.display_image(FRAME_BUFFER_OUTPUT)
            return
        
        alg = self.algorithm_combo.currentText()
        thr = self.threshold_slider.value()
        contrast_factor = self.contrast_slider.value() / 100.0
        pixel_s = self.scale_slider.value()
        
        print(f"Applying {alg} with threshold {thr}, contrast {contrast_factor:.2f}, scale {pixel_s}")
        
        # Check if we need to convert between RGB and grayscale
        needs_grayscale = not self.rgb_mode.isChecked()
        is_currently_grayscale = (len(FRAME_BUFFER_ORIGINAL.shape) == 2)
        
        # Handle transition between color modes
        if needs_grayscale and not is_currently_grayscale:
            # Convert RGB to grayscale
            if FRAME_BUFFER_GRAYSCALE is None or FRAME_BUFFER_GRAYSCALE.shape != (FRAME_BUFFER_ORIGINAL.shape[0], FRAME_BUFFER_ORIGINAL.shape[1]):
                FRAME_BUFFER_GRAYSCALE = np.empty((FRAME_BUFFER_ORIGINAL.shape[0], FRAME_BUFFER_ORIGINAL.shape[1]), dtype=np.uint8)
            
            # Calculate grayscale with weighted average
            FRAME_BUFFER_GRAYSCALE = np.dot(FRAME_BUFFER_ORIGINAL[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # Reuse the global processing buffer with correct shape
            FRAME_BUFFER_PROCESSING = np.empty_like(FRAME_BUFFER_GRAYSCALE)
            np.copyto(FRAME_BUFFER_PROCESSING, FRAME_BUFFER_GRAYSCALE)
        elif not needs_grayscale and is_currently_grayscale:
            # Cannot convert grayscale to RGB (would need the original), just use grayscale as all channels
            print("Warning: Cannot convert from grayscale to RGB, using grayscale for all channels")
            rgb_from_gray = np.stack([FRAME_BUFFER_ORIGINAL] * 3, axis=-1)
            
            # Reuse the global processing buffer with correct shape
            if FRAME_BUFFER_PROCESSING is None or FRAME_BUFFER_PROCESSING.shape != rgb_from_gray.shape:
                FRAME_BUFFER_PROCESSING = np.empty_like(rgb_from_gray)
            np.copyto(FRAME_BUFFER_PROCESSING, rgb_from_gray)
        else:
            # No color mode change, use the original directly
            if FRAME_BUFFER_PROCESSING is None or FRAME_BUFFER_PROCESSING.shape != FRAME_BUFFER_ORIGINAL.shape:
                FRAME_BUFFER_PROCESSING = np.empty_like(FRAME_BUFFER_ORIGINAL)
            np.copyto(FRAME_BUFFER_PROCESSING, FRAME_BUFFER_ORIGINAL)
        
        # Apply contrast if needed
        if abs(contrast_factor - 1.0) > 0.01:
            try:
                # Apply contrast directly to NumPy array
                if len(FRAME_BUFFER_PROCESSING.shape) == 3:  # RGB
                    # Apply to each channel separately
                    for c in range(3):
                        channel = FRAME_BUFFER_PROCESSING[:,:,c].astype(np.float32)
                        # Simple contrast adjustment formula: f(x) = 128 + contrast_factor * (x - 128)
                        channel = 128 + contrast_factor * (channel - 128)
                        FRAME_BUFFER_PROCESSING[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
                else:  # Grayscale
                    FRAME_BUFFER_PROCESSING = FRAME_BUFFER_PROCESSING.astype(np.float32)
                    FRAME_BUFFER_PROCESSING = 128 + contrast_factor * (FRAME_BUFFER_PROCESSING - 128)
                    FRAME_BUFFER_PROCESSING = np.clip(FRAME_BUFFER_PROCESSING, 0, 255).astype(np.uint8)
                print("Applied contrast adjustment.")
            except Exception as e:
                print(f"Error applying contrast: {e}")
                import traceback
                traceback.print_exc()
        
        # Apply selected dithering algorithm
        if alg == "Floyd-Steinberg":
            result_array = self.floyd_steinberg_numpy(FRAME_BUFFER_PROCESSING, thr, pixel_s)
        elif alg == "Simple Threshold":
            result_array = self.simple_threshold(FRAME_BUFFER_PROCESSING, thr, pixel_s)
        else:
            result_array = None
        
        # Store result in global output buffer
        if result_array is not None:
            if FRAME_BUFFER_OUTPUT is None or FRAME_BUFFER_OUTPUT.shape != result_array.shape:
                FRAME_BUFFER_OUTPUT = np.empty_like(result_array)
            np.copyto(FRAME_BUFFER_OUTPUT, result_array)
        
        # Update toggle button state
            self.toggle_button.setEnabled(True)
            
            # If not showing original, update display with new dithered image
            if not self.showing_original:
                self.display_image(FRAME_BUFFER_OUTPUT)
        else:
            print("Error: Dithering result is None")
    
    def save_image(self):
        """Save the current dithered image to a file"""
        global FRAME_BUFFER_OUTPUT
        
        # Check if we have a dithered image to save
        if FRAME_BUFFER_OUTPUT is None:
            print("No image to save")
            return
        
        try:
            # Get save path
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
            )
            
            # Handle case where user canceled the dialog
            if not path:
                print("Save canceled by user")
                return
            
            # Ensure path has a valid extension
            if not (path.lower().endswith('.png') or path.lower().endswith('.jpg') or 
                    path.lower().endswith('.jpeg')):
                path += '.png'
            
            print(f"Saving image to {path}, shape={FRAME_BUFFER_OUTPUT.shape}")
            
            # Use OpenCV for saving - simple and direct
            import cv2
            
            # Make a copy to avoid modifying the original
            save_array = FRAME_BUFFER_OUTPUT.copy()
            
            # Convert RGB to BGR for OpenCV if it's a color image
            if len(save_array.shape) == 3 and save_array.shape[2] == 3:
                save_array = save_array[:, :, ::-1]  # RGB to BGR conversion
            
            # Save the image
            cv2.imwrite(path, save_array)
            print(f"Image saved successfully to {path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()
    
    def floyd_steinberg_numpy(self, array, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
            
        # Access shared algorithm buffer
        global SHARED_ALGORITHM_BUFFER
        
        if self.rgb_mode.isChecked():
            mode = 'RGB'
        else:
            mode = 'L'
            
        if pixel_scale == 1:
            try:
                # Ensure shared buffer is allocated and correct size/shape
                if SHARED_ALGORITHM_BUFFER is None or SHARED_ALGORITHM_BUFFER.shape != array.shape:
                    # Buffer doesn't exist or has wrong shape, create a new one
                    SHARED_ALGORITHM_BUFFER = np.empty_like(array, dtype=np.float32)
                    print(f"Created new Floyd-Steinberg buffer with shape {array.shape}")
                elif SHARED_ALGORITHM_BUFFER.dtype != np.float32:
                    # Buffer exists but wrong type, recreate
                    SHARED_ALGORITHM_BUFFER = np.empty_like(array, dtype=np.float32)
                    print(f"Recreated Floyd-Steinberg buffer with correct dtype")
                
                # Copy input to shared buffer
                np.copyto(SHARED_ALGORITHM_BUFFER, array)
                
                # Apply dithering to shared buffer
                result = fs_dither(SHARED_ALGORITHM_BUFFER, mode, threshold)
                
                # Return result directly - it's the same buffer
                return result
            except Exception as e:
                print(f"Error in shared buffer handling (falling back to normal mode): {e}")
                # Fall back to normal mode if there's an error with the shared buffer
                arr = array.astype(np.float32)
                arr = fs_dither(arr, mode, threshold)
                return arr.astype(np.uint8)
        else:
            # For pixel scaling, use existing implementation
            arr = array.astype(np.uint8)
            result_arr = downscale_dither_upscale(arr, threshold, pixel_scale, mode)
            return result_arr
    
    def simple_threshold(self, array, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
            
        # Access shared algorithm buffer
        global SHARED_ALGORITHM_BUFFER
        
        if self.rgb_mode.isChecked():
            type = 'RGB'
        else:
            type = 'L'
            
        if pixel_scale == 1:
            try:
                # Ensure shared buffer is allocated and correct size/shape
                if SHARED_ALGORITHM_BUFFER is None or SHARED_ALGORITHM_BUFFER.shape != array.shape:
                    # Buffer doesn't exist or has wrong shape, create a new one
                    SHARED_ALGORITHM_BUFFER = np.empty_like(array, dtype=np.uint8)
                    print(f"Created new Simple Threshold buffer with shape {array.shape}")
                elif SHARED_ALGORITHM_BUFFER.dtype != np.uint8:
                    # Buffer exists but wrong type, recreate
                    SHARED_ALGORITHM_BUFFER = np.empty_like(array, dtype=np.uint8)
                    print(f"Recreated Simple Threshold buffer with correct dtype")
                
                # Copy input to shared buffer
                np.copyto(SHARED_ALGORITHM_BUFFER, array)
                    
                if type == 'RGB':
                    # Process directly in shared buffer
                    result = simple_threshold_rgb_ps1(SHARED_ALGORITHM_BUFFER, threshold)
                    return result
                else:
                    # For grayscale, use vectorized operations on shared buffer
                    np.where(SHARED_ALGORITHM_BUFFER < threshold, 0, 255, out=SHARED_ALGORITHM_BUFFER)
                    return SHARED_ALGORITHM_BUFFER
            except Exception as e:
                print(f"Error in shared buffer handling (falling back to normal mode): {e}")
                # Fall back to normal method if there's a shared buffer error
                img_array = array.astype(np.uint8)
                if type == 'RGB':
                    result = simple_threshold_rgb_ps1(img_array, threshold)
                    return result.astype(np.uint8)
                else:
                    result = np.where(img_array < threshold, 0, 255).astype(np.uint8)
                    return result
        else:
            # For pixel scaling, use existing implementation
            orig_h, orig_w = array.shape[:2]
            out_array = simple_threshold_dither(array, type, pixel_scale, orig_w, orig_h, threshold)
            return out_array

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
            
        # Accept the event and close the application
        print("Cleanup complete, closing application")
        event.accept()

    def pass_through_changed(self):
        """Called when the Pass-through checkbox state changes"""
        print(f"Pass-through mode changed to: {self.pass_through_mode.isChecked()}")
        
        # If in camera mode, immediately apply the change
        if self.camera_mode_active:
            print("Applying pass-through mode change to camera feed")
            
        # For static images, apply the change if auto-render is on
        if self.auto_render.isChecked() and FRAME_BUFFER_ORIGINAL is not None:
            if self.pass_through_mode.isChecked():
                # In pass-through mode, display original image
                self.display_image(FRAME_BUFFER_ORIGINAL)
                self.showing_original = True
                self.toggle_button.setText("Switch to Dithered Image")
            else:
                # Apply dithering when exiting pass-through mode
                self.apply_dither()
                self.showing_original = False
                self.toggle_button.setText("Switch to Original Image")

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
    
    # Set initial values
    FRAME_BUFFER_ORIGINAL = None
    FRAME_BUFFER_PROCESSING = None
    FRAME_BUFFER_OUTPUT = None
    FRAME_BUFFER_GRAYSCALE = None
    SHARED_ALGORITHM_BUFFER = None
    LAST_FRAME_TIME = time_module.time()
    FRAME_COUNT = 0
    
    # Try to ensure clean camera state at application startup
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

    app = QApplication(sys.argv)
    window = DitherApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
