import sys
import pygame
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
SHARED_ALGORITHM_BUFFER = None   # Shared buffer for all algorithms - prevents memory allocation lag
DOWNSCALED_BUFFER_RGB = None     # Shared buffer for downscaled RGB images
DOWNSCALED_BUFFER_GRAY = None    # Shared buffer for downscaled grayscale images

# Frame handling metrics
LAST_FRAME_TIME = 0              # Time when last frame was processed
LAST_FPS = 0                     # Last calculated FPS
FRAME_COUNT = 0                  # Frame counter for FPS calculation
FRAME_PROCESS_TIME = 0           # Time spent processing the last frame

# Initialize pygame
pygame.init()

# Set up the display
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 320
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(config.APP_NAME)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Font
font = pygame.font.SysFont(None, 24)

# Button and Slider definitions
class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.active = False

    def draw(self, surface):
        color = GREEN if self.active else GRAY
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.callback = callback
        self.active = False

    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        pos = int(self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(surface, RED, (pos, self.rect.centery), 5)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.active = False
        elif event.type == pygame.MOUSEMOTION and self.active:
            rel_x = event.pos[0] - self.rect.x
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            self.value = max(self.min_val, min(self.max_val, self.value))
            self.callback(self.value)
            return True
        return False

# Define buttons and sliders
buttons = []
sliders = []

# Example button: Apply Dither
def apply_dither_callback():
    # Call the apply_dither function from the original code
    apply_dither()

buttons.append(Button(10, 10, 100, 30, "Apply Dither", apply_dither_callback))

# Example slider: Threshold
def threshold_callback(value):
    # Update threshold value
    print(f"Threshold updated: {value}")

sliders.append(Slider(10, 50, 200, 20, 1, 254, 128, threshold_callback))

# Main loop
running = True
clock = pygame.time.Clock()

def apply_dither():
    """Apply dithering to the current image"""
    global FRAME_BUFFER_ORIGINAL
    global FRAME_BUFFER_OUTPUT
    global DOWNSCALED_BUFFER_RGB
    global DOWNSCALED_BUFFER_GRAY

    if FRAME_BUFFER_ORIGINAL is None:
        return  # No image to process

    try:
        # Get current settings
        alg = "Floyd-Steinberg"  # Default algorithm
        thr = 128  # Default threshold
        contrast_factor = 1.0  # Default contrast
        pixel_s = 1  # Default pixel scale

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
            if True:  # RGB mode
                # Process for RGB mode
                if alg == "Floyd-Steinberg":
                    result = fs_dither(work_copy.astype(np.float32), 'RGB', thr)
                elif alg == "Bayer":
                    result = bayer_dither(work_copy, 'RGB', thr)
                else:  # Simple Threshold
                    result = simple_threshold_rgb_ps1(work_copy, thr)
            else:
                # Process for Grayscale
                # Convert to grayscale first
                gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

                if alg == "Floyd-Steinberg":
                    result = fs_dither(gray.astype(np.float32), 'L', thr)
                elif alg == "Bayer":
                    result = bayer_dither(gray, 'L', thr)
                else:  # Simple Threshold
                    result = np.where(gray < thr, 0, 255).astype(np.uint8)
        else:
            # Process with downscaling and upscaling for pixel_s > 1
            if True:  # RGB mode
                # RGB mode
                if alg == "Floyd-Steinberg":
                    # This function already handles the complete pipeline
                    result = downscale_dither_upscale(work_copy, thr, pixel_s, 'RGB')
                elif alg == "Bayer":
                    # For Bayer, first downscale, then apply bayer dithering, then upscale
                    # Step 1: Downscale
                    small_h = max(1, orig_h // pixel_s)
                    small_w = max(1, orig_w // pixel_s)

                    if True:  # RGB mode
                        # Ensure the shared downscaled buffer has the right dimensions
                        if DOWNSCALED_BUFFER_RGB is None or DOWNSCALED_BUFFER_RGB.shape[:2] != (small_h, small_w):
                            DOWNSCALED_BUFFER_RGB = np.empty((small_h, small_w, 3), dtype=np.float32)

                        # RGB block averaging using the shared buffer
                        small_arr = block_average_rgb(work_copy, DOWNSCALED_BUFFER_RGB, small_h, small_w, pixel_s)

                        # Step 2: Apply Bayer dithering to downscaled image
                        small_result = bayer_dither(small_arr, 'RGB', thr)

                        # Step 3: Upscale back to original size
                        upscaled = np.empty((work_copy.shape[0], work_copy.shape[1], 3), dtype=np.uint8)
                        result = nearest_upscale_rgb(small_result, upscaled, work_copy.shape[0], work_copy.shape[1], small_h, small_w, pixel_s)
                    else:
                        # Ensure the shared downscaled buffer has the right dimensions
                        if DOWNSCALED_BUFFER_GRAY is None or DOWNSCALED_BUFFER_GRAY.shape != (small_h, small_w):
                            DOWNSCALED_BUFFER_GRAY = np.empty((small_h, small_w), dtype=np.float32)

                        # Grayscale block averaging using the shared buffer
                        small_arr = block_average_gray(work_copy, DOWNSCALED_BUFFER_GRAY, small_h, small_w, pixel_s)

                        # Apply Bayer dithering to downscaled image
                        small_result = bayer_dither(small_arr, 'L', thr)

                        # Upscale back to original size
                        upscaled = np.empty((work_copy.shape[0], work_copy.shape[1]), dtype=np.uint8)
                        result = nearest_upscale_gray(small_result, upscaled, work_copy.shape[0], work_copy.shape[1], small_h, small_w, pixel_s)
                else:  # Simple Threshold
                    result = simple_threshold_rgb_ps1(work_copy, thr)
            else:
                # Grayscale mode
                # Convert to grayscale first
                gray = np.dot(work_copy[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

                if alg == "Floyd-Steinberg":
                    # This function already handles the complete pipeline
                    result = downscale_dither_upscale(gray, thr, pixel_s, 'L')
                elif alg == "Bayer":
                    # For Bayer, first downscale, then apply bayer dithering, then upscale
                    # Step 1: Downscale
                    small_h = max(1, gray.shape[0] // pixel_s)
                    small_w = max(1, gray.shape[1] // pixel_s)

                    # Ensure the shared downscaled buffer has the right dimensions
                    if DOWNSCALED_BUFFER_GRAY is None or DOWNSCALED_BUFFER_GRAY.shape != (small_h, small_w):
                        DOWNSCALED_BUFFER_GRAY = np.empty((small_h, small_w), dtype=np.float32)

                    # Grayscale block averaging using the shared buffer
                    small_arr = block_average_gray(gray, DOWNSCALED_BUFFER_GRAY, small_h, small_w, pixel_s)

                    # Apply Bayer dithering to downscaled image
                    small_result = bayer_dither(small_arr, 'L', thr)

                    # Upscale back to original size
                    upscaled = np.empty((gray.shape[0], gray.shape[1]), dtype=np.uint8)
                    result = nearest_upscale_gray(small_result, upscaled, gray.shape[0], gray.shape[1], small_h, small_w, pixel_s)
                else:  # Simple Threshold
                    result = np.where(gray < thr, 0, 255).astype(np.uint8)

        # Ensure the result has the same shape as the original
        if True and result.shape != orig_shape:  # RGB mode
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

    except Exception as e:
        print(f"Error applying dither: {e}")
        import traceback
        traceback.print_exc()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for button in buttons:
            button.handle_event(event)
        for slider in sliders:
            slider.handle_event(event)
        # Handle keyboard events for slider updates
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                # Increase threshold slider value
                for slider in sliders:
                    if slider.rect.collidepoint(pygame.mouse.get_pos()):
                        slider.value = min(slider.max_val, slider.value + 1)
                        slider.callback(slider.value)
            elif event.key == pygame.K_DOWN:
                # Decrease threshold slider value
                for slider in sliders:
                    if slider.rect.collidepoint(pygame.mouse.get_pos()):
                        slider.value = max(slider.min_val, slider.value - 1)
                        slider.callback(slider.value)

    # Clear the screen
    screen.fill(WHITE)

    # Draw buttons and sliders
    for button in buttons:
        button.draw(screen)
    for slider in sliders:
        slider.draw(screen)

    # Display the output buffer if available
    if FRAME_BUFFER_OUTPUT is not None:
        # Convert NumPy array to Pygame surface
        output_surface = pygame.surfarray.make_surface(FRAME_BUFFER_OUTPUT)
        screen.blit(output_surface, (0, 0))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit() 