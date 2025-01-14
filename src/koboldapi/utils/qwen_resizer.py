"""
This module contains image and video resizing algorithms adapted from Qwen-VL.
https://github.com/QwenLM/Qwen2-VL/

Adapted portions Copyright 2024 Alibaba Cloud
Licensed under the Apache License, Version 2.0
"""

import math
from typing import Tuple

# Constants from Qwen2-VL
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """ Round number to nearest multiple of factor """
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """ Round up to nearest multiple of factor """
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """ Round down to nearest multiple of factor """
    return math.floor(number / factor) * factor

def qwen_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                min_pixels: int = VIDEO_MIN_PIXELS,
                max_pixels: int = VIDEO_MAX_PIXELS) -> Tuple[int, int]:
    """ Qwen-VL's smart resize algorithm.
    
        Args:
            height: Original image height
            width: Original image width
            factor: Resize factor (default: 28)
            min_pixels: Minimum total pixels
            max_pixels: Maximum total pixels
            
        Returns:
            Tuple of (new_height, new_width)
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def qwen_frame_count(total_frames: int, video_fps: float, 
                    fps_max_frames: int) -> int:
    """ Calculate optimal number of frames using Qwen-VL algorithm.
    
        Args:
            total_frames: Total frames in video
            video_fps: Video frames per second
            fps_max_frames: Maximum frames to extract
            
        Returns:
            Number of frames to extract
    """
    nframes = total_frames / video_fps * FPS
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(fps_max_frames, total_frames), FRAME_FACTOR)
    nframes = min(max(nframes, min_frames), max_frames)
    return round_by_factor(nframes, FRAME_FACTOR)