#!/usr/bin/env python3
"""
Benchmark script to measure CPU usage and processing time
for the dithering operations.
"""

import time
import numpy as np
from PIL import Image
import psutil
import helper
import argparse

def benchmark_threshold(image_array, iterations=100, pixel_scale=1, threshold=128):
    """
    Benchmark simple threshold performance.
    """
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=0.1)  # Reset CPU measurement
    
    mode = 'RGB' if len(image_array.shape) == 3 else 'L'
    
    for _ in range(iterations):
        if mode == 'RGB':
            if pixel_scale == 1:
                helper.simple_threshold_rgb_ps1(image_array, threshold)
            else:
                orig_h, orig_w = image_array.shape[:2]
                helper.simple_threshold_rgb_psMore(image_array, pixel_scale, orig_w, orig_h, threshold)
        else:
            if pixel_scale == 1:
                np.where(image_array < threshold, 0, 255).astype(np.uint8)
            else:
                orig_h, orig_w = image_array.shape[:2]
                helper.simple_threshold_greyscale_psMore(image_array, pixel_scale, orig_w, orig_h, threshold)
    
    cpu_usage = psutil.cpu_percent(interval=None)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'cpu_usage': cpu_usage
    }

def benchmark_fs_dither(image_array, iterations=100, pixel_scale=1, threshold=128):
    """
    Benchmark Floyd-Steinberg dithering performance.
    """
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=0.1)  # Reset CPU measurement
    
    mode = 'RGB' if len(image_array.shape) == 3 else 'L'
    
    # Convert to float32 for dithering
    array_float = image_array.astype(np.float32)
    
    for _ in range(iterations):
        helper.fs_dither(array_float.copy(), mode, threshold)
    
    cpu_usage = psutil.cpu_percent(interval=None)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'cpu_usage': cpu_usage
    }

def run_benchmarks(image_path, iterations=50):
    """
    Run a series of benchmarks on the given image.
    """
    print(f"Loading image: {image_path}")
    # Load image in both RGB and grayscale
    img_rgb = Image.open(image_path).convert('RGB')
    img_gray = Image.open(image_path).convert('L')
    
    # Convert to NumPy arrays
    rgb_array = np.array(img_rgb)
    gray_array = np.array(img_gray)
    
    print(f"Image dimensions: {rgb_array.shape}")
    print(f"Running {iterations} iterations for each test...")
    
    results = []
    
    # RGB Simple Threshold (Pixel Scale 1)
    print("\nBenchmarking RGB Simple Threshold (Pixel Scale 1)...")
    result = benchmark_threshold(rgb_array, iterations, pixel_scale=1)
    print(f"  Total time: {result['total_time']:.3f}s")
    print(f"  Average time per iteration: {result['avg_time']*1000:.2f}ms")
    print(f"  CPU usage: {result['cpu_usage']:.1f}%")
    results.append(("RGB Simple Threshold (PS=1)", result))
    
    # RGB Simple Threshold (Pixel Scale 4)
    print("\nBenchmarking RGB Simple Threshold (Pixel Scale 4)...")
    result = benchmark_threshold(rgb_array, iterations, pixel_scale=4)
    print(f"  Total time: {result['total_time']:.3f}s")
    print(f"  Average time per iteration: {result['avg_time']*1000:.2f}ms")
    print(f"  CPU usage: {result['cpu_usage']:.1f}%")
    results.append(("RGB Simple Threshold (PS=4)", result))
    
    # Grayscale Simple Threshold (Pixel Scale 1)
    print("\nBenchmarking Grayscale Simple Threshold (Pixel Scale 1)...")
    result = benchmark_threshold(gray_array, iterations, pixel_scale=1)
    print(f"  Total time: {result['total_time']:.3f}s")
    print(f"  Average time per iteration: {result['avg_time']*1000:.2f}ms")
    print(f"  CPU usage: {result['cpu_usage']:.1f}%")
    results.append(("Grayscale Simple Threshold (PS=1)", result))
    
    # RGB Floyd-Steinberg
    print("\nBenchmarking RGB Floyd-Steinberg...")
    result = benchmark_fs_dither(rgb_array, iterations)
    print(f"  Total time: {result['total_time']:.3f}s")
    print(f"  Average time per iteration: {result['avg_time']*1000:.2f}ms")
    print(f"  CPU usage: {result['cpu_usage']:.1f}%")
    results.append(("RGB Floyd-Steinberg", result))
    
    # Grayscale Floyd-Steinberg
    print("\nBenchmarking Grayscale Floyd-Steinberg...")
    result = benchmark_fs_dither(gray_array, iterations)
    print(f"  Total time: {result['total_time']:.3f}s")
    print(f"  Average time per iteration: {result['avg_time']*1000:.2f}ms")
    print(f"  CPU usage: {result['cpu_usage']:.1f}%")
    results.append(("Grayscale Floyd-Steinberg", result))
    
    print("\n=== SUMMARY ===")
    for name, result in results:
        print(f"{name}: {result['avg_time']*1000:.2f}ms, CPU: {result['cpu_usage']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark dithering algorithms')
    parser.add_argument('--image', type=str, default='example.png', help='Path to image file')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    
    args = parser.parse_args()
    
    run_benchmarks(args.image, args.iterations) 