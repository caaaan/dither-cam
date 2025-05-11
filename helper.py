from numba import njit, prange
import numpy as np

@njit
def fs_dither_rgb(img_array, threshold=128):
    """Optimized Floyd-Steinberg dithering for RGB images"""
    h, w, c = img_array.shape
    
    # Create error buffers for each color channel
    error_r = np.zeros((h, w+2), dtype=np.float32)
    error_g = np.zeros((h, w+2), dtype=np.float32)
    error_b = np.zeros((h, w+2), dtype=np.float32)
    
    # Copy original values to error buffers
    for y in range(h):
        for x in range(w):
            error_r[y, x+1] = float(img_array[y, x, 0])
            error_g[y, x+1] = float(img_array[y, x, 1])
            error_b[y, x+1] = float(img_array[y, x, 2])
    
    # Process the image row by row
    for y in range(h):
        for x in range(w):
            # Get current values from error buffers
            old_r = error_r[y, x+1]
            old_g = error_g[y, x+1]
            old_b = error_b[y, x+1]
            
            # Apply threshold to get new binary values
            new_r = 0.0 if old_r < threshold else 255.0
            new_g = 0.0 if old_g < threshold else 255.0
            new_b = 0.0 if old_b < threshold else 255.0
            
            # Store results in output
            img_array[y, x, 0] = int(new_r)
            img_array[y, x, 1] = int(new_g)
            img_array[y, x, 2] = int(new_b)
            
            # Calculate errors
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # Distribute errors to neighboring pixels
            error_r[y, x+2] += err_r * 7/16        # right
            error_g[y, x+2] += err_g * 7/16
            error_b[y, x+2] += err_b * 7/16
            
            if y+1 < h:
                error_r[y+1, x] += err_r * 3/16    # bottom left
                error_g[y+1, x] += err_g * 3/16
                error_b[y+1, x] += err_b * 3/16
                
                error_r[y+1, x+1] += err_r * 5/16  # bottom
                error_g[y+1, x+1] += err_g * 5/16
                error_b[y+1, x+1] += err_b * 5/16
                
                error_r[y+1, x+2] += err_r * 1/16  # bottom right
                error_g[y+1, x+2] += err_g * 1/16
                error_b[y+1, x+2] += err_b * 1/16
    
    return img_array

@njit
def fs_dither_greyscale(img_array, threshold=128):
    """Optimized Floyd-Steinberg dithering for grayscale images"""
    h, w = img_array.shape
    
    # Create error buffer array to avoid modifying original values too early
    # This prevents artifacts in the dithering pattern
    error_buffer = np.zeros((h, w+2), dtype=np.float32)  # +2 width for boundary handling
    
    # Copy original values to error buffer
    for y in range(h):
        for x in range(w):
            error_buffer[y, x+1] = float(img_array[y, x])  # +1 offset
    
    # Process the image row by row
    for y in range(h):
        for x in range(w):
            # Get current value from error buffer
            old_val = error_buffer[y, x+1]  # +1 offset
            
            # Apply threshold to get new binary value
            new_val = 0.0 if old_val < threshold else 255.0
            
            # Store result in output
            img_array[y, x] = int(new_val)
            
            # Calculate error
            error = old_val - new_val
            
            # Distribute error to neighboring pixels (error diffusion)
            error_buffer[y, x+2] += error * 7/16        # right
            if y+1 < h:
                error_buffer[y+1, x] += error * 3/16    # bottom left
                error_buffer[y+1, x+1] += error * 5/16  # bottom
                error_buffer[y+1, x+2] += error * 1/16  # bottom right
    
    return img_array

def fs_dither(arr, type, threshold=128):
    """Apply Floyd-Steinberg dithering to an array"""
    # Make a copy of the input array to avoid modifying the original
    work_arr = arr.copy()
    
    if type == 'RGB':
        result = fs_dither_rgb(work_arr, threshold)
    else:  # type == 'L'
        result = fs_dither_greyscale(work_arr, threshold)
        
    # Ensure final result is uint8
    return np.clip(result, 0, 255).astype(np.uint8)

# Optimized version for small grayscale images without parallelization overhead
@njit
def fs_dither_greyscale_small(img_array, threshold=128):
    h, w = img_array.shape
    for y in range(h):
        for x in range(w):
            old = img_array[y, x]
            new = 0 if old < threshold else 255
            err = old - new
            img_array[y, x] = new
            if x+1 < w:
                img_array[y, x+1] += err * 7/16
            if y+1 < h:
                if x > 0: img_array[y+1, x-1] += err * 3/16
                img_array[y+1, x] += err * 5/16
                if x+1 < w: img_array[y+1, x+1] += err * 1/16
    return np.clip(img_array, 0, 255).astype(np.uint8)

# Optimized version for small RGB images without parallelization overhead
@njit
def fs_dither_rgb_small(img_array, threshold=128):
    h, w, c = img_array.shape
    for y in range(h):
        for x in range(w):
            old_r = img_array[y, x, 0]
            old_g = img_array[y, x, 1]
            old_b = img_array[y, x, 2]
            
            new_r = 0 if old_r < threshold else 255
            new_g = 0 if old_g < threshold else 255
            new_b = 0 if old_b < threshold else 255
            
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            img_array[y, x, 0] = new_r
            img_array[y, x, 1] = new_g
            img_array[y, x, 2] = new_b
            
            if x+1 < w:
                img_array[y, x+1, 0] += err_r * 7/16
                img_array[y, x+1, 1] += err_g * 7/16
                img_array[y, x+1, 2] += err_b * 7/16
            if y+1 < h:
                if x > 0:
                    img_array[y+1, x-1, 0] += err_r * 3/16
                    img_array[y+1, x-1, 1] += err_g * 3/16
                    img_array[y+1, x-1, 2] += err_b * 3/16
                
                img_array[y+1, x, 0] += err_r * 5/16
                img_array[y+1, x, 1] += err_g * 5/16
                img_array[y+1, x, 2] += err_b * 5/16
                
                if x+1 < w:
                    img_array[y+1, x+1, 0] += err_r * 1/16
                    img_array[y+1, x+1, 1] += err_g * 1/16
                    img_array[y+1, x+1, 2] += err_b * 1/16
    return np.clip(img_array, 0, 255).astype(np.uint8)
    
@njit
def simple_threshold_rgb_ps1(arr, threshold=128):
    """Vectorized threshold for RGB images with pixel scale 1"""
    # Fast direct implementation for small arrays
    if arr.size < 10000:  # For small images
        result = np.zeros_like(arr)
        h, w, c = arr.shape
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    result[y, x, c] = 0 if arr[y, x, c] < threshold else 255
        return result
    else:
        # Use vectorized operations for larger images
        result = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            result[:, :, c] = np.where(arr[:, :, c] < threshold, 0, 255)
        return result

@njit(parallel=True)
def simple_threshold_greyscale_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    """Optimized threshold for grayscale images with pixel scale > 1"""
    # For small images, don't create the block lookup arrays
    if orig_h * orig_w < 10000:
        out_array = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for y0 in range(0, orig_h, pixel_scale):
            y1 = min(y0 + pixel_scale, orig_h)
            for x0 in range(0, orig_w, pixel_scale):
                x1 = min(x0 + pixel_scale, orig_w)
                
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                
                block = img_array[y0:y1, x0:x1]
                mean_val = np.mean(block)
                block_color = 0 if mean_val < threshold else 255
                out_array[y0:y1, x0:x1] = block_color
        return out_array
    
    # For larger images, use the more complex optimization
    # Pre-allocate output array
    out_array = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Pre-calculate block boundaries to avoid redundant calculations
    y_blocks = [(y, min(y + pixel_scale, orig_h)) 
               for y in range(0, orig_h, pixel_scale)]
    
    x_blocks = [(x, min(x + pixel_scale, orig_w)) 
               for x in range(0, orig_w, pixel_scale)]
    
    # Process blocks in parallel
    for y_idx in prange(len(y_blocks)):
        y0, y1 = y_blocks[y_idx]
        for x_idx in range(len(x_blocks)):
            x0, x1 = x_blocks[x_idx]
            
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            
            # Calculate block mean
            block = img_array[y0:y1, x0:x1]
            mean_val = np.mean(block)
            
            # Apply threshold and fill the block
            block_color = 0 if mean_val < threshold else 255
            out_array[y0:y1, x0:x1] = block_color
    
    return out_array

@njit(parallel=True)
def _simple_threshold_rgb_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    """Optimized threshold for RGB images with pixel scale > 1 - internal implementation"""
    # For small images, use a simpler approach
    if orig_h * orig_w < 10000:
        out_array = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        means = np.zeros(3, dtype=np.float32)
        
        for y0 in range(0, orig_h, pixel_scale):
            y1 = min(y0 + pixel_scale, orig_h)
            for x0 in range(0, orig_w, pixel_scale):
                x1 = min(x0 + pixel_scale, orig_w)
                
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                
                block = img_array[y0:y1, x0:x1, :]
                for c in range(3):
                    channel_block = block[:, :, c]
                    means[c] = np.mean(channel_block)
                
                block_color = np.where(means < threshold, 0, 255).astype(np.uint8)
                for c in range(3):
                    out_array[y0:y1, x0:x1, c] = block_color[c]
        return out_array
    
    # For larger images, use the more complex optimization
    # Pre-allocate output array
    out_array = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    
    # Pre-calculate block boundaries to avoid redundant calculations
    y_blocks = [(y, min(y + pixel_scale, orig_h)) 
               for y in range(0, orig_h, pixel_scale)]
    
    x_blocks = [(x, min(x + pixel_scale, orig_w)) 
               for x in range(0, orig_w, pixel_scale)]
    
    # Pre-allocate means array
    means = np.zeros(3, dtype=np.float32)
    
    # Process blocks in parallel
    for y_idx in prange(len(y_blocks)):
        y0, y1 = y_blocks[y_idx]
        for x_idx in range(len(x_blocks)):
            x0, x1 = x_blocks[x_idx]
            
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            
            # Calculate mean for each channel
            block = img_array[y0:y1, x0:x1, :]
            for c in range(3):
                channel_block = block[:, :, c]
                means[c] = np.mean(channel_block)
            
            # Apply threshold to each channel
            block_color = np.where(means < threshold, 0, 255).astype(np.uint8)
            
            # Fill the block with the calculated color
            for c in range(3):
                out_array[y0:y1, x0:x1, c] = block_color[c]
    
    return out_array

def simple_threshold_rgb_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    """Wrapper function that ensures img_array is 3D before passing to JIT-compiled function"""
    # Check dimensions and convert if needed
    if len(img_array.shape) != 3:
        # Create a 3D array from the 2D grayscale (outside the JIT compiled function)
        print(f"Converting 2D grayscale array of shape {img_array.shape} to 3D for RGB processing")
        img_array_3d = np.stack([img_array] * 3, axis=-1)
        return _simple_threshold_rgb_psMore(img_array_3d, pixel_scale, orig_w, orig_h, threshold)
    else:
        # Already 3D, pass through to JIT function
        return _simple_threshold_rgb_psMore(img_array, pixel_scale, orig_w, orig_h, threshold)

def simple_threshold_dither(arr, type, pixel_scale, orig_w, orig_h, threshold=128):
    """Direct thresholding with specified pixel scale"""
    if type == 'RGB':
        return simple_threshold_rgb_psMore(arr, pixel_scale, orig_w, orig_h, threshold)
    else:
        return simple_threshold_greyscale_psMore(arr, pixel_scale, orig_w, orig_h, threshold)

@njit
def block_average_rgb(array, out_array, small_h, small_w, pixel_s):
    """Optimized block averaging for RGB images with Numba"""
    orig_h, orig_w, _ = array.shape
    
    for y in range(small_h):
        y_start = y * pixel_s
        y_end = min((y+1) * pixel_s, orig_h)
        for x in range(small_w):
            x_start = x * pixel_s
            x_end = min((x+1) * pixel_s, orig_w)
            
            # Sum values for each channel
            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0
            count = 0
            
            # Manual averaging is faster with Numba than using np.mean with axis parameter
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    r_sum += array[yy, xx, 0]
                    g_sum += array[yy, xx, 1]
                    b_sum += array[yy, xx, 2]
                    count += 1
            
            if count > 0:
                out_array[y, x, 0] = r_sum / count
                out_array[y, x, 1] = g_sum / count
                out_array[y, x, 2] = b_sum / count
    
    return out_array

@njit
def block_average_gray(array, out_array, small_h, small_w, pixel_s):
    """Optimized block averaging for grayscale images with Numba"""
    orig_h, orig_w = array.shape
    
    for y in range(small_h):
        y_start = y * pixel_s
        y_end = min((y+1) * pixel_s, orig_h)
        for x in range(small_w):
            x_start = x * pixel_s
            x_end = min((x+1) * pixel_s, orig_w)
            
            # Manual sum is faster with Numba
            val_sum = 0.0
            count = 0
            
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    val_sum += array[yy, xx]
                    count += 1
            
            if count > 0:
                out_array[y, x] = val_sum / count
    
    return out_array

@njit
def nearest_upscale_rgb(small_arr, out_array, orig_h, orig_w, small_h, small_w, pixel_s):
    """Optimized nearest neighbor upscaling for RGB images"""
    for y in range(orig_h):
        y_small = min(y // pixel_s, small_h-1)
        for x in range(orig_w):
            x_small = min(x // pixel_s, small_w-1)
            out_array[y, x, 0] = small_arr[y_small, x_small, 0]
            out_array[y, x, 1] = small_arr[y_small, x_small, 1]
            out_array[y, x, 2] = small_arr[y_small, x_small, 2]
    return out_array

@njit
def nearest_upscale_gray(small_arr, out_array, orig_h, orig_w, small_h, small_w, pixel_s):
    """Optimized nearest neighbor upscaling for grayscale images"""
    for y in range(orig_h):
        y_small = min(y // pixel_s, small_h-1)
        for x in range(orig_w):
            x_small = min(x // pixel_s, small_w-1)
            out_array[y, x] = small_arr[y_small, x_small]
    return out_array

@njit
def downscale_dither_upscale_rgb(array, threshold, pixel_scale):
    """Complete pipeline for downscale-dither-upscale for RGB images with JIT optimization"""
    # Get dimensions
    orig_h, orig_w, _ = array.shape
    small_h = max(1, orig_h // pixel_scale)
    small_w = max(1, orig_w // pixel_scale)
    
    # Step 1: Downscale using block averaging
    small_arr = np.zeros((small_h, small_w, 3), dtype=np.float32)
    
    # Optimized block averaging
    for y in range(small_h):
        y_start = y * pixel_scale
        y_end = min((y+1) * pixel_scale, orig_h)
        for x in range(small_w):
            x_start = x * pixel_scale
            x_end = min((x+1) * pixel_scale, orig_w)
            
            # For each channel, average the block
            r_sum, g_sum, b_sum, count = 0.0, 0.0, 0.0, 0
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    r_sum += array[yy, xx, 0]
                    g_sum += array[yy, xx, 1]
                    b_sum += array[yy, xx, 2]
                    count += 1
                    
            if count > 0:
                small_arr[y, x, 0] = r_sum / count
                small_arr[y, x, 1] = g_sum / count
                small_arr[y, x, 2] = b_sum / count
    
    # Step 2: Apply dithering on smaller array
    # We'll use an optimized version of fs_dither_rgb directly
    error_r = np.zeros((small_h, small_w+2), dtype=np.float32)
    error_g = np.zeros((small_h, small_w+2), dtype=np.float32)
    error_b = np.zeros((small_h, small_w+2), dtype=np.float32)
    
    # Copy small array values to error buffers
    for y in range(small_h):
        for x in range(small_w):
            error_r[y, x+1] = small_arr[y, x, 0]
            error_g[y, x+1] = small_arr[y, x, 1]
            error_b[y, x+1] = small_arr[y, x, 2]
    
    # Dither the small array
    for y in range(small_h):
        for x in range(small_w):
            # Get current values from error buffers
            old_r = error_r[y, x+1]
            old_g = error_g[y, x+1]
            old_b = error_b[y, x+1]
            
            # Apply threshold to get binary values
            new_r = 0.0 if old_r < threshold else 255.0
            new_g = 0.0 if old_g < threshold else 255.0
            new_b = 0.0 if old_b < threshold else 255.0
            
            # Store results in small array
            small_arr[y, x, 0] = new_r
            small_arr[y, x, 1] = new_g
            small_arr[y, x, 2] = new_b
            
            # Calculate error
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # Distribute error to neighboring pixels
            error_r[y, x+2] += err_r * 7/16        # right
            error_g[y, x+2] += err_g * 7/16
            error_b[y, x+2] += err_b * 7/16
            
            if y+1 < small_h:
                error_r[y+1, x] += err_r * 3/16    # bottom left
                error_g[y+1, x] += err_g * 3/16
                error_b[y+1, x] += err_b * 3/16
                
                error_r[y+1, x+1] += err_r * 5/16  # bottom
                error_g[y+1, x+1] += err_g * 5/16
                error_b[y+1, x+1] += err_b * 5/16
                
                error_r[y+1, x+2] += err_r * 1/16  # bottom right
                error_g[y+1, x+2] += err_g * 1/16
                error_b[y+1, x+2] += err_b * 1/16
    
    # Step 3: Upscale using nearest neighbor
    result = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    
    # Optimization: Use a faster algorithm for small images
    if small_h * small_w < 10000:
        for y in range(orig_h):
            y_small = min(y // pixel_scale, small_h-1)
            for x in range(orig_w):
                x_small = min(x // pixel_scale, small_w-1)
                result[y, x, 0] = small_arr[y_small, x_small, 0]
                result[y, x, 1] = small_arr[y_small, x_small, 1]
                result[y, x, 2] = small_arr[y_small, x_small, 2]
    else:
        # For larger images, we can use the same approach
        for y in range(orig_h):
            y_small = min(y // pixel_scale, small_h-1)
            for x in range(orig_w):
                x_small = min(x // pixel_scale, small_w-1)
                result[y, x, 0] = small_arr[y_small, x_small, 0]
                result[y, x, 1] = small_arr[y_small, x_small, 1]
                result[y, x, 2] = small_arr[y_small, x_small, 2]
    
    return result

@njit
def downscale_dither_upscale_gray(array, threshold, pixel_scale):
    """Complete pipeline for downscale-dither-upscale for grayscale images with JIT optimization"""
    # Get dimensions
    orig_h, orig_w = array.shape
    small_h = max(1, orig_h // pixel_scale)
    small_w = max(1, orig_w // pixel_scale)
    
    # Step 1: Downscale using block averaging
    small_arr = np.zeros((small_h, small_w), dtype=np.float32)
    
    # Optimized block averaging
    for y in range(small_h):
        y_start = y * pixel_scale
        y_end = min((y+1) * pixel_scale, orig_h)
        for x in range(small_w):
            x_start = x * pixel_scale
            x_end = min((x+1) * pixel_scale, orig_w)
            
            # Average the block
            val_sum, count = 0.0, 0
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    val_sum += array[yy, xx]
                    count += 1
                    
            if count > 0:
                small_arr[y, x] = val_sum / count
    
    # Step 2: Apply dithering on smaller array
    # We'll use an optimized version of fs_dither_greyscale directly
    error_buffer = np.zeros((small_h, small_w+2), dtype=np.float32)
    
    # Copy small array values to error buffer
    for y in range(small_h):
        for x in range(small_w):
            error_buffer[y, x+1] = small_arr[y, x]
    
    # Dither the small array
    for y in range(small_h):
        for x in range(small_w):
            # Get current value from error buffer
            old_val = error_buffer[y, x+1]
            
            # Apply threshold to get binary value
            new_val = 0.0 if old_val < threshold else 255.0
            
            # Store result in small array
            small_arr[y, x] = new_val
            
            # Calculate error
            error = old_val - new_val
            
            # Distribute error to neighboring pixels
            error_buffer[y, x+2] += error * 7/16        # right
            if y+1 < small_h:
                error_buffer[y+1, x] += error * 3/16    # bottom left
                error_buffer[y+1, x+1] += error * 5/16  # bottom
                error_buffer[y+1, x+2] += error * 1/16  # bottom right
    
    # Step 3: Upscale using nearest neighbor
    result = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Upscale using direct assignment
    for y in range(orig_h):
        y_small = min(y // pixel_scale, small_h-1)
        for x in range(orig_w):
            x_small = min(x // pixel_scale, small_w-1)
            result[y, x] = small_arr[y_small, x_small]
    
    return result

def downscale_dither_upscale(array, threshold, pixel_scale, mode='RGB'):
    """Unified function for the complete downscale-dither-upscale pipeline with mode selection"""
    if mode == 'RGB':
        return downscale_dither_upscale_rgb(array, threshold, pixel_scale)
    else:  # mode == 'L'
        return downscale_dither_upscale_gray(array, threshold, pixel_scale)

@njit
def bgr_to_rgb(array):
    """Convert BGR color format to RGB format using Numba acceleration"""
    # Simple color channel swap
    return array[:, :, ::-1].copy()

@njit
def optimized_pass_through(array, out_buffer=None):
    """
    Optimized function for pass-through mode that minimizes memory operations.
    This function only performs necessary color corrections while avoiding unnecessary memory allocations.
    
    Parameters:
    array: Input array to process
    out_buffer: Optional pre-allocated output buffer to reuse for better performance
    
    Returns:
    Processed array (either in out_buffer or a new array)
    """
    if out_buffer is not None and out_buffer.shape == array.shape:
        # Reuse existing buffer for better performance
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                for c in range(array.shape[2]):
                    out_buffer[y, x, c] = array[y, x, c]
        return out_buffer
    else:
        # Just make a clean copy if no buffer available
        return array.copy()

# Bayer Matrix dithering implementation
@njit
def get_bayer_matrix(size):
    """Generate a Bayer matrix of given size (must be power of 2)"""
    if size == 2:
        # Base 2x2 Bayer matrix
        return np.array([[0, 2], 
                         [3, 1]], dtype=np.uint8)
    else:
        # Recursively build larger matrices
        m = get_bayer_matrix(size // 2)
        return np.block([[4*m, 4*m+2], 
                          [4*m+3, 4*m+1]])

@njit
def bayer_dither_rgb(img_array, threshold=128, matrix_size=8):
    """Apply Bayer dithering to RGB images"""
    h, w, c = img_array.shape
    
    # Generate or get the Bayer matrix
    bayer_matrix = get_bayer_matrix(matrix_size)
    matrix_h, matrix_w = bayer_matrix.shape
    
    # Normalize the matrix to range 0-255
    normalized_matrix = (bayer_matrix * 255) // (matrix_size * matrix_size)
    
    # Apply dithering
    for y in range(h):
        for x in range(w):
            # Get matrix value for this pixel
            matrix_val = normalized_matrix[y % matrix_h, x % matrix_w]
            
            # Apply to each channel
            for ch in range(3):
                pixel_val = img_array[y, x, ch]
                # Adjust threshold by matrix value - creates the dithering pattern
                # Values higher than matrix offset stay white, others become black
                img_array[y, x, ch] = 0 if pixel_val < (threshold + matrix_val - 128) else 255
    
    return img_array

@njit
def bayer_dither_greyscale(img_array, threshold=128, matrix_size=8):
    """Apply Bayer dithering to grayscale images"""
    h, w = img_array.shape
    
    # Generate or get the Bayer matrix
    bayer_matrix = get_bayer_matrix(matrix_size)
    matrix_h, matrix_w = bayer_matrix.shape
    
    # Normalize the matrix to range 0-255
    normalized_matrix = (bayer_matrix * 255) // (matrix_size * matrix_size)
    
    # Apply dithering
    for y in range(h):
        for x in range(w):
            # Get matrix value for this pixel
            matrix_val = normalized_matrix[y % matrix_h, x % matrix_w]
            
            # Apply to pixel
            pixel_val = img_array[y, x]
            img_array[y, x] = 0 if pixel_val < (threshold + matrix_val - 128) else 255
    
    return img_array

def bayer_dither(arr, type, threshold=128):
    """Apply Bayer dithering to an array"""
    # Make a copy of the input array to avoid modifying the original
    work_arr = arr.copy()
    matrix_size = 8  # 8x8 Bayer matrix (can be adjusted)
    
    if type == 'RGB':
        result = bayer_dither_rgb(work_arr, threshold, matrix_size)
    else:  # type == 'L'
        result = bayer_dither_greyscale(work_arr, threshold, matrix_size)
        
    # Ensure final result is uint8
    return np.clip(result, 0, 255).astype(np.uint8)