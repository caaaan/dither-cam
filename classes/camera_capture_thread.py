import threading
import time
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
import numpy as np
import gc
import os
import traceback

# Import helper functions from the classes folder
from classes.helper import (fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither, 
                           block_average_rgb, block_average_gray, nearest_upscale_rgb, 
                           nearest_upscale_gray, downscale_dither_upscale)

# Try to import picamera only if available (for development on non-Pi platforms)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera module not available. Camera features will be disabled.")

class CameraCaptureThread(QThread):
    frameProcessed = pyqtSignal(Image.Image)  # Emit processed frames directly

    def __init__(self, app_instance):
        super().__init__()
        self.is_running = False  # Start as not running
        self.app = app_instance  # Reference to main app for processing settings
        self.camera = None
        self.camera_initialized = False
        self.camera_lock = threading.Lock()  # Add lock for thread safety
        
        # Add buffer reuse to reduce memory allocation
        self.frame_buffer = None  # Will be initialized on first frame capture
        self.gray_buffer = None   # For grayscale conversions
        
        # Simple timing control
        self.last_capture_time = 0
        self.min_capture_interval = 0.033  # ~30fps maximum (33ms)
        
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
        self.target_cpu_percent = 45  # Target CPU usage percentage
        self.last_cpu_check = time.time()
        self.cpu_check_interval = 1.0  # Check CPU usage every second
        
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
                
                # Clear old buffers to force recreation with new size
                self.frame_buffer = None
                self.gray_buffer = None
                
                return True
                
        except Exception as e:
            print(f"Error reconfiguring camera: {e}")
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
                        
                        # Test by capturing one frame - if this succeeds, camera is working
                        test_frame = self.camera.capture_array()
                        if test_frame is not None:
                            print(f"Test frame captured successfully: {test_frame.shape}")
                            self.camera_initialized = True
                            print("Camera initialized successfully")
                            break
                        else:
                            raise RuntimeError("Test frame capture returned None")
                            
                except Exception as e:
                    print(f"Camera initialization attempt {attempt+1} failed: {e}")
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
                        
                        # Reuse buffer if possible
                        if self.frame_buffer is None or self.frame_buffer.shape != frame.shape:
                            self.frame_buffer = np.empty_like(frame)
                        np.copyto(self.frame_buffer, frame)
                        
                        # Process based on mode
                        if self.app.rgb_mode.isChecked():
                            # RGB mode
                            processed_array = self.process_frame_array(self.frame_buffer, 'RGB')
                        else:
                            # Create grayscale conversion
                            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                            
                            # Process grayscale
                            processed_array = self.process_frame_array(gray, 'L')
                        
                        # Convert to PIL image for display
                        if processed_array is not None:
                            if processed_array.ndim == 3:
                                pil_result = Image.fromarray(processed_array, 'RGB')
                            else:
                                pil_result = Image.fromarray(processed_array, 'L')
                                
                            # Send to UI
                            self.frameProcessed.emit(pil_result)
                    else:
                        print(f"Invalid frame format: {frame.shape if frame is not None else None}")
                    
                    # Update timing and adjust capture rate if needed
                    process_end = time.time()
                    process_time = process_end - process_start
                    
                    # Set minimum interval to slightly more than processing time
                    # This ensures we don't capture frames faster than we can process
                    self.min_capture_interval = process_time * 1.1
                    
                    # Make sure we don't go below 30fps or above reasonable limits
                    self.min_capture_interval = max(0.033, min(0.2, self.min_capture_interval))
                    
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
                    traceback.print_exc()
                    # Exit on device errors, continue for other errors
                    if "device" in str(e).lower() or "resource" in str(e).lower():
                        print("Device error detected, exiting capture loop")
                        break
                
        except Exception as e:
            print(f"Critical error in camera thread: {e}")
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
            
            # Check if we have a small image
            is_small = array.size < 150000  # Adjust based on your threshold
            
            # Optimize copy operations for small images
            if is_small:
                # For small images, direct copy is faster than empty_like + copyto
                array_to_dither = array.copy()
            else:
                # For larger images, use the existing optimized approach
                if mode == 'RGB':
                    array_to_dither = np.empty_like(array)
                    np.copyto(array_to_dither, array)
                else:
                    array_to_dither = np.empty_like(array)
                    np.copyto(array_to_dither, array)
            
            # Apply contrast if needed
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
            
            # Apply selected dithering algorithm directly on NumPy array
            if alg == "Floyd-Steinberg":
                if pixel_s == 1:
                    # Apply dithering directly
                    if mode == 'RGB':
                        result = fs_dither(array_to_dither.astype(np.float32), 'RGB', thr)
                    else:
                        result = fs_dither(array_to_dither.astype(np.float32), 'L', thr)
                else:
                    # Use the optimized downscale-dither-upscale pipeline
                    result = downscale_dither_upscale(array_to_dither, thr, pixel_s, mode)
            elif alg == "Simple Threshold":
                if pixel_s == 1:
                    # Apply simple threshold directly
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
                    traceback.print_exc()
                    self.camera = None
                    self.camera_initialized = False
            else:
                print("Camera was already None")
        
        # Try to help the system release camera resources
        try:
            # Run garbage collection to help release resources
            gc.collect()
            time.sleep(0.5)
            
            # Try to force system-level cleanup as well
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