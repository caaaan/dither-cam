import sys
import cv2
import numpy as np
import os
import config
from helper import (fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither, 
                   block_average_rgb, block_average_gray, nearest_upscale_rgb, 
                   nearest_upscale_gray, downscale_dither_upscale, bgr_to_rgb, 
                   optimized_pass_through, bayer_dither)

# Global frame handling variables
FRAME_BUFFER_ORIGINAL = None     # Original frame buffer (RGB format)
FRAME_BUFFER_PROCESSING = None   # Buffer used during processing steps
FRAME_BUFFER_OUTPUT = None       # Final frame after processing (to display)
FRAME_BUFFER_GRAYSCALE = None    # Grayscale version when needed
SHARED_ALGORITHM_BUFFER = None   # Shared buffer for all algorithms
DOWNSCALED_BUFFER_RGB = None     # Shared buffer for downscaled RGB images
DOWNSCALED_BUFFER_GRAY = None    # Shared buffer for downscaled grayscale images

# Global settings
current_algorithm = "Floyd-Steinberg"
current_threshold = 128
current_contrast = 100
current_scale = 1
rgb_mode = True
pass_through = False

def nothing(x):
    """Callback function for trackbars"""
    pass

def create_control_window():
    """Create a window for controls"""
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 300, 400)
    
    # Create trackbars
    cv2.createTrackbar('Threshold', 'Controls', current_threshold, 254, nothing)
    cv2.createTrackbar('Contrast', 'Controls', current_contrast, 200, nothing)
    cv2.createTrackbar('Scale', 'Controls', current_scale, 8, nothing)
    
    # Create buttons (using trackbars as buttons)
    cv2.createTrackbar('Algorithm', 'Controls', 0, 2, nothing)  # 0: Floyd-Steinberg, 1: Bayer, 2: Simple
    cv2.createTrackbar('RGB Mode', 'Controls', 1, 1, nothing)   # 0: Grayscale, 1: RGB
    cv2.createTrackbar('Pass Through', 'Controls', 0, 1, nothing)  # 0: Off, 1: On

def get_current_settings():
    """Get current settings from trackbars"""
    global current_algorithm, current_threshold, current_contrast, current_scale, rgb_mode, pass_through
    
    # Get values from trackbars
    current_threshold = cv2.getTrackbarPos('Threshold', 'Controls')
    current_contrast = cv2.getTrackbarPos('Contrast', 'Controls') / 100.0
    current_scale = cv2.getTrackbarPos('Scale', 'Controls')
    
    # Get algorithm
    alg_pos = cv2.getTrackbarPos('Algorithm', 'Controls')
    algorithms = ["Floyd-Steinberg", "Bayer", "Simple Threshold"]
    current_algorithm = algorithms[alg_pos]
    
    # Get modes
    rgb_mode = bool(cv2.getTrackbarPos('RGB Mode', 'Controls'))
    pass_through = bool(cv2.getTrackbarPos('Pass Through', 'Controls'))

def apply_dither():
    """Apply dithering to the current image"""
    global FRAME_BUFFER_ORIGINAL, FRAME_BUFFER_OUTPUT, DOWNSCALED_BUFFER_RGB, DOWNSCALED_BUFFER_GRAY
    
    if FRAME_BUFFER_ORIGINAL is None:
        return  # No image to process
        
    try:
        # Get current settings
        get_current_settings()
        
        if pass_through:
            # Just copy the original to output
            FRAME_BUFFER_OUTPUT = FRAME_BUFFER_ORIGINAL.copy()
            return
            
        # Create a copy of the original for processing
        work_copy = FRAME_BUFFER_ORIGINAL.copy()
        orig_shape = work_copy.shape
        orig_h, orig_w = orig_shape[:2]
        
        # Apply contrast adjustment if needed
        if abs(current_contrast - 1.0) > 0.01:
            work_copy = work_copy.astype(np.float32)
            work_copy = 128 + current_contrast * (work_copy - 128)
            work_copy = np.clip(work_copy, 0, 255).astype(np.uint8)
        
        # Apply dithering based on mode and pixel scale
        if current_scale == 1:
            # Process at original resolution
            if rgb_mode:
                if current_algorithm == "Floyd-Steinberg":
                    result = fs_dither(work_copy.astype(np.float32), 'RGB', current_threshold)
                elif current_algorithm == "Bayer":
                    result = bayer_dither(work_copy, 'RGB', current_threshold)
                else:  # Simple Threshold
                    result = simple_threshold_rgb_ps1(work_copy, current_threshold)
            else:
                # Convert to grayscale
                gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                
                if current_algorithm == "Floyd-Steinberg":
                    result = fs_dither(gray.astype(np.float32), 'L', current_threshold)
                elif current_algorithm == "Bayer":
                    result = bayer_dither(gray, 'L', current_threshold)
                else:  # Simple Threshold
                    result = np.where(gray < current_threshold, 0, 255).astype(np.uint8)
        else:
            # Process with downscaling and upscaling
            if rgb_mode:
                if current_algorithm == "Floyd-Steinberg":
                    result = downscale_dither_upscale(work_copy, current_threshold, current_scale, 'RGB')
                elif current_algorithm == "Bayer":
                    small_h = max(1, orig_h // current_scale)
                    small_w = max(1, orig_w // current_scale)
                    
                    if DOWNSCALED_BUFFER_RGB is None or DOWNSCALED_BUFFER_RGB.shape[:2] != (small_h, small_w):
                        DOWNSCALED_BUFFER_RGB = np.empty((small_h, small_w, 3), dtype=np.float32)
                    
                    small_arr = block_average_rgb(work_copy, DOWNSCALED_BUFFER_RGB, small_h, small_w, current_scale)
                    small_result = bayer_dither(small_arr, 'RGB', current_threshold)
                    
                    upscaled = np.empty((work_copy.shape[0], work_copy.shape[1], 3), dtype=np.uint8)
                    result = nearest_upscale_rgb(small_result, upscaled, work_copy.shape[0], work_copy.shape[1], small_h, small_w, current_scale)
                else:  # Simple Threshold
                    result = simple_threshold_rgb_ps1(work_copy, current_threshold)
            else:
                # Grayscale mode
                gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                
                if current_algorithm == "Floyd-Steinberg":
                    result = downscale_dither_upscale(gray, current_threshold, current_scale, 'L')
                elif current_algorithm == "Bayer":
                    small_h = max(1, gray.shape[0] // current_scale)
                    small_w = max(1, gray.shape[1] // current_scale)
                    
                    if DOWNSCALED_BUFFER_GRAY is None or DOWNSCALED_BUFFER_GRAY.shape != (small_h, small_w):
                        DOWNSCALED_BUFFER_GRAY = np.empty((small_h, small_w), dtype=np.float32)
                    
                    small_arr = block_average_gray(gray, DOWNSCALED_BUFFER_GRAY, small_h, small_w, current_scale)
                    small_result = bayer_dither(small_arr, 'L', current_threshold)
                    
                    upscaled = np.empty((gray.shape[0], gray.shape[1]), dtype=np.uint8)
                    result = nearest_upscale_gray(small_result, upscaled, gray.shape[0], gray.shape[1], small_h, small_w, current_scale)
                else:  # Simple Threshold
                    result = np.where(gray < current_threshold, 0, 255).astype(np.uint8)
        
        # Ensure the result has the same shape as the original
        if rgb_mode and result.shape != orig_shape:
            if len(result.shape) == 2 and len(orig_shape) == 3:
                rgb_result = np.empty(orig_shape, dtype=np.uint8)
                for c in range(3):
                    rgb_result[:,:,c] = result
                result = rgb_result
        
        # Update output buffer
        FRAME_BUFFER_OUTPUT = result
        
    except Exception as e:
        print(f"Error applying dither: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    global FRAME_BUFFER_ORIGINAL, FRAME_BUFFER_OUTPUT
    
    # Create control window
    create_control_window()
    
    # Create main window
    cv2.namedWindow('Dither Cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Dither Cam', 480, 320)
    
    # Load a test image if available
    try:
        test_img = cv2.imread("ditherer.jpeg")
        if test_img is not None:
            FRAME_BUFFER_ORIGINAL = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            apply_dither()
    except Exception as e:
        print(f"Error loading test image: {e}")
    
    while True:
        # Get current settings
        get_current_settings()
        
        # Apply dithering
        apply_dither()
        
        # Display the output if available
        if FRAME_BUFFER_OUTPUT is not None:
            # Convert RGB to BGR for OpenCV display
            display_img = cv2.cvtColor(FRAME_BUFFER_OUTPUT, cv2.COLOR_RGB2BGR)
            cv2.imshow('Dither Cam', display_img)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('o'):  # Open image
            file_path = cv2.FileDialog.getOpenFileName(None, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")[0]
            if file_path:
                img = cv2.imread(file_path)
                if img is not None:
                    FRAME_BUFFER_ORIGINAL = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    apply_dither()
        elif key == ord('s'):  # Save image
            if FRAME_BUFFER_OUTPUT is not None:
                file_path = cv2.FileDialog.getSaveFileName(None, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")[0]
                if file_path:
                    save_img = cv2.cvtColor(FRAME_BUFFER_OUTPUT, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_path, save_img)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 