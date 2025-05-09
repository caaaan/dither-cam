from numba import njit
import numpy as np

@njit
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

@njit
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

    # Process each band separately for RGB using NumPy
    
    result = np.zeros_like(arr)
    
    # Apply threshold to each channel independently
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        result[:, :, c] = np.where(channel < threshold, 0, 255)
    
    return result

@njit
def simple_threshold_greyscale_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    out_array = np.zeros_like(img_array)
            
    for y0 in range(0, orig_h, pixel_scale):
        for x0 in range(0, orig_w, pixel_scale):
            x1 = min(x0 + pixel_scale, orig_w)
            y1 = min(y0 + pixel_scale, orig_h)
            
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            

            block = img_array[y0:y1, x0:x1]
            # Calculate mean
            mean = np.mean(block)
            # Apply threshold
            block_color = 0 if mean < threshold else 255
            # Fill the output block
            out_array[y0:y1, x0:x1] = block_color
    return out_array

@njit
def simple_threshold_rgb_psMore(img_array, pixel_scale, orig_w, orig_h, threshold=128):
    out_array = np.zeros_like(img_array)
            
    for y0 in range(0, orig_h, pixel_scale):
        for x0 in range(0, orig_w, pixel_scale):
            x1 = min(x0 + pixel_scale, orig_w)
            y1 = min(y0 + pixel_scale, orig_h)
            
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue

            block = img_array[y0:y1, x0:x1, :]
            # Calculate mean for each channel manually instead of using axis=(0,1)
            means = np.zeros(3, dtype=np.float32)
            for c in range(3):  # Assuming 3 channels (RGB)
                channel_block = block[:, :, c]
                means[c] = np.mean(channel_block)  # This uses axis=None which is supported
            
            # Apply threshold to each channel
            block_color = np.where(means < threshold, 0, 255).astype(np.uint8)
            # Fill the output block
            out_array[y0:y1, x0:x1, :] = block_color
            
    return out_array

def simple_threshold_dither(arr, type, pixel_scale, orig_w, orig_h, threshold=128):
    if type == 'RGB':
        return simple_threshold_rgb_psMore(arr, pixel_scale, orig_w, orig_h, threshold)
    else:
        return simple_threshold_greyscale_psMore(arr, pixel_scale, orig_w, orig_h, threshold)