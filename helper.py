from numba import njit, prange
import numpy as np

@njit(parallel=True)
def fs_dither_rgb(img_array, threshold=128):
    h, w, c = img_array.shape
    for y in range(h):
        for x in range(w):
            # Access pixel values directly without copying
            old_r = img_array[y, x, 0]
            old_g = img_array[y, x, 1]
            old_b = img_array[y, x, 2]
            
            # Calculate new values
            new_r = 0 if old_r < threshold else 255
            new_g = 0 if old_g < threshold else 255
            new_b = 0 if old_b < threshold else 255
            
            # Calculate errors
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # Set the new pixel values
            img_array[y, x, 0] = new_r
            img_array[y, x, 1] = new_g
            img_array[y, x, 2] = new_b
            
            # Distribute errors to neighboring pixels
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

@njit(parallel=True)
def fs_dither_greyscale(img_array, threshold=128):
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

def fs_dither(arr, type, threshold=128):
    # Make a copy of the input array to avoid modifying the original
    work_arr = arr.copy()
    
    if type == 'RGB':
        return fs_dither_rgb(work_arr, threshold)
    else:  # type == 'L'
        return fs_dither_greyscale(work_arr, threshold)
    
@njit
def simple_threshold_rgb_ps1(arr, threshold=128):
    """Vectorized threshold for RGB images with pixel scale 1"""
    # Use vectorized operations for better performance
    result = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        result[:, :, c] = np.where(arr[:, :, c] < threshold, 0, 255)
    return result

@njit(parallel=True)
def simple_threshold_greyscale_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    """Optimized threshold for grayscale images with pixel scale > 1"""
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
def simple_threshold_rgb_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    """Optimized threshold for RGB images with pixel scale > 1"""
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

def simple_threshold_dither(arr, type, pixel_scale, orig_w, orig_h, threshold=128):
    """Direct thresholding with specified pixel scale"""
    if type == 'RGB':
        return simple_threshold_rgb_psMore(arr, pixel_scale, orig_w, orig_h, threshold)
    else:
        return simple_threshold_greyscale_psMore(arr, pixel_scale, orig_w, orig_h, threshold)