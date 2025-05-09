import time
import queue
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
import numpy as np

# Import helper functions from the classes folder
from classes.helper import (fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither, 
                           downscale_dither_upscale)

class FrameProcessingThread(QThread):
    frameProcessed = pyqtSignal(Image.Image)

    def __init__(self, frame_queue, app_instance):
        super().__init__()
        self.is_running = False  # Start as not running
        self.frame_queue = frame_queue
        self.app = app_instance  # Reference to the main app for dithering settings
        print("Processing thread initialized")
        
        # Buffer reuse for reduced memory allocation
        self.pil_buffer = None
        self.result_buffer = None
        self.rgb_buffer = None
        self.gray_buffer = None
        
        # Thread synchronization - detect if we're falling behind
        self.processed_count = 0
        self.queue_full_count = 0
        self.last_process_time = 0
        self.avg_process_time = 0.03  # Initial guess: 30ms per frame

    def run(self):
        print("Frame processing thread starting...")
        self.is_running = True
        
        # Performance tracking
        frames_processed = 0
        last_report_time = time.time()
        
        # Reusable buffers
        rgb_np_buffer = None
        gray_np_buffer = None
        
        while self.is_running:
            try:
                # Use timeout to ensure we regularly check if thread should stop
                try:
                    # Non-blocking queue get with timeout for better responsiveness
                    frame = self.frame_queue.get(timeout=0.1)
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
                
                # Make sure we have a proper RGB frame
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    try:
                        # Convert to grayscale directly in NumPy if needed
                        if not self.app.rgb_mode.isChecked():
                            # Create or reuse grayscale buffer
                            if gray_np_buffer is None or gray_np_buffer.shape[:2] != frame.shape[:2]:
                                # Grayscale conversion directly in NumPy
                                gray_np_buffer = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                            else:
                                # Reuse buffer, just update values
                                np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140], out=gray_np_buffer)
                            
                            # Process grayscale NumPy array directly
                            processed_array = self.process_frame_array(gray_np_buffer, 'L')
                        else:
                            # Process RGB NumPy array directly
                            processed_array = self.process_frame_array(frame, 'RGB')
                        
                        if processed_array is not None:
                            # Only convert to PIL at the very end for display
                            if processed_array.ndim == 3:
                                # RGB array
                                pil_result = Image.fromarray(processed_array, 'RGB')
                            else:
                                # Grayscale array
                                pil_result = Image.fromarray(processed_array, 'L')
                            
                            # Emit processed frame - only if we're still running
                            if self.is_running:
                                self.frameProcessed.emit(pil_result)
                        
                        # Update processing time statistics for adaptive timing
                        process_end = time.time()
                        process_time = process_end - process_start
                        
                        # Exponential moving average for process time
                        self.avg_process_time = 0.9 * self.avg_process_time + 0.1 * process_time
                        
                        # Report FPS every 5 seconds
                        self.processed_count += 1
                        current_time = time.time()
                        if current_time - last_report_time > 5.0:
                            fps = self.processed_count / (current_time - last_report_time)
                            print(f"Frame processing rate: {fps:.1f} FPS, avg process time: {self.avg_process_time*1000:.1f}ms")
                            self.processed_count = 0
                            last_report_time = current_time
                            
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        traceback.print_exc()
                else:
                    print(f"Invalid frame format: {frame.shape if frame is not None else None}")
            except Exception as e:
                print(f"Error in processing thread main loop: {e}")
                traceback.print_exc()
                # Don't exit the loop for occasional errors
        
        print("Processing thread stopped")

    def process_frame(self, pil_img):
        """Apply dithering to a frame based on current app settings"""
        try:
            # Get current settings from main app
            alg = self.app.algorithm_combo.currentText()
            thr = self.app.threshold_slider.value()
            contrast_factor = self.app.contrast_slider.value() / 100.0
            pixel_s = self.app.scale_slider.value()
            use_rgb = self.app.rgb_mode.isChecked()
            
            # Prepare the image format as needed
            if use_rgb and pil_img.mode != 'RGB':
                # Convert to RGB format
                image_to_dither = pil_img.convert('RGB')
            elif not use_rgb and pil_img.mode != 'L':
                # Convert to grayscale format
                image_to_dither = pil_img.convert('L')
            else:
                # Already in the right format
                image_to_dither = pil_img
            
            # Apply contrast if needed
            if abs(contrast_factor - 1.0) > 0.01:
                try:
                    enhancer = ImageEnhance.Contrast(image_to_dither)
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
            traceback.print_exc()
            return None

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

    def stop(self):
        """Stop the processing thread safely"""
        print("Requesting processing thread to stop...")
        self.is_running = False 