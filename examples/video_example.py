"""
This file contains code adapted from Qwen-VL 
https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

Adapted portions Copyright 2024 Alibaba Cloud
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this adapted code except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

The following components are adapted from Qwen-VL:
- Core resizing algorithms (smart_resize, smart_nframes)
- Utility functions (round_by_factor, ceil_by_factor, floor_by_factor)
- Key constants and parameters for video processing

All other code Copyright 2024 github/jabberjabberjabber 
Licensed under GNU General Public License v3.0
"""

import decord
import base64
import io
import math
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from koboldapi import KoboldAPICore
from PIL import Image

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
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                min_pixels: int = VIDEO_MIN_PIXELS, 
                max_pixels: int = VIDEO_MAX_PIXELS) -> tuple[int, int]:
    """Smart resize algorithm from Qwen2-VL."""
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

def smart_nframes(total_frames: int, video_fps: float, fps_max_frames: int) -> int:
    """Calculate number of frames to extract based on Qwen2-VL algorithm."""
    nframes = total_frames / video_fps * FPS
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(fps_max_frames, total_frames), FRAME_FACTOR)
    nframes = min(max(nframes, min_frames), max_frames)
    return round_by_factor(nframes, FRAME_FACTOR)
    
def process_video(video_path: str | Path, max_frames: int) -> list[str]:
    """
        Process video according to Qwen2-VL specifications and return base64 encoded frames.
        Frames are taken in intervals according to the smart_nframes algorithm.
        Each interval is 8 frames .5 seconds apart for a total of 4 seconds.
        Frames are then resized according to the smart_resize algorithm.
        Args:
            video_path: Path to video file (str or Path)
            
        Returns:
            List of base64 encoded JPEG frames in temporal order
    """
    video_path = str(video_path)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    n_frames = smart_nframes(total_frames, video_fps, fps_max_frames=max_frames)
    video_length = total_frames // video_fps
    indices = np.linspace(0, total_frames - 1, n_frames).round().astype(int).tolist()
    
    # Ensure even number of frames, duplicate one if not
    if len(indices) % 2 != 0:
        indices.append(indices[-1])

    frames = vr.get_batch(indices).asnumpy()
    height, width = frames.shape[1:3]
    min_pixels = VIDEO_MIN_PIXELS 
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / n_frames * FRAME_FACTOR), 
                    int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    base64_frames = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        base64_frames.append(img_str)
    
    return base64_frames, video_length
    
def analyze_video(video_path: str, api_url: str, templates: str,
                 max_frames: int = 64, output_dir: str = None,
                 batch_size: int = 8) -> dict:
    """ Analyze an entire video by sending batches of frames to a 
        Koboldcpp API and get a rolling summary.

        Args:
            video_path: Path to video file
            api_url: URL of Kobold API endpoint
            template_dir: Directory containing prompt templates
            max_frames: Maximum number of frames to process
            output_dir: Directory to save results (defaults to video location)
            batch_size: Number of frames to process in each batch
            
        Returns:
            dict: Contains frame analysis and final summary
    """
    
    config_dict = {
        "api_url": api_url,
        "api_password": "",  
        "templates_directory": templates,
        "min_p": 0,
        "rep_pen": 1,
        "temperature": 0.1,
        "top_k": 0,
        "top_p": 1
    }
    core = KoboldAPICore(config_dict)
    video_path = Path(video_path)
    max_context = core.api_client.get_max_context_length()
    out_path = Path(output_dir) if output_dir else video_path.parent / f"{video_path.stem}_analysis"
    out_path.mkdir(exist_ok=True)
    
    results = {
        "analysis": [],
        "final_summary": None,
        "metadata": {
            "video_path": str(video_path.absolute()),
            "api_url": api_url,
            "max_frames": max_frames,
            "batch_size": batch_size,
        }
    }
    print(f"Extracting frames from {video_path}...")
    frames, total_length = process_video(video_path, max_frames)
    frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    results["metadata"]["frame_count"] = len(frames)
    total_batches = len(frame_batches)
    for batch_idx, frame_batch in enumerate(frame_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        batch_analysis = analyze_frame_batch(
            client=core.api_client,
            wrapper=core.template_wrapper,
            frame_batch=frame_batch,
            batch_idx=batch_idx,
            batch_size=batch_size,
            total_length=total_length
        )
        results["analysis"].append({
            "batch": batch_idx + 1,
            "analysis": batch_analysis
        })
    if frame_batches:
        final_summary = generate_final_summary(
            client=core.api_client,
            wrapper=core.template_wrapper,
            results=results,
            total_length=total_length
        )
        results["final_summary"] = final_summary
    results_file = out_path / "analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results
    
def analyze_frame_batch(client, wrapper, frame_batch, batch_idx, batch_size, total_length):
    """ Analyze a batch of frames using the Kobold API.
    """
    frame_instruction = f"These are frames a half second apart from a video. The time between the first and last frame is four seconds. The total length of the video is {total_length}s. Out of {batch_size} groups evenly divided in linear time, this frame group is number {batch_idx}. Describe the the action occuring over this time period. Pay close attention to characters, style, movement and scene. Make assumptions as needed using knowledge of common video content, themes, and tropes."
    frame_system = "You are Qwen, a helpful assistant made by Alibaba with video processing capability."
    frame_prompt = wrapper.wrap_prompt(frame_instruction, "", frame_system)[0]
    return client.generate(
        prompt=frame_prompt,
        images=frame_batch,
        max_length=500,
        min_p=0,
        rep_pen=1,
        temperature=0.1,
        top_k=0,
        top_p=1
    )

def generate_final_summary(client, wrapper, results, total_length):
    """ Generate comprehensive final summary using all analyses.
    """
    max_context = client.get_max_context_length()
    all_analyses = "\n\n".join(
        f"Group: {analysis['batch']}, Events: {analysis['analysis']}" 
        for analysis in results["analysis"]
        )
    final_instruction = (f"Recall the events in a linear sequence to create a description of the entire video. Use as much of the individual descriptions as possible."
    )
    final_content = f"{all_analyses}"
    final_system = "You are Qwen, a helpful assistant made by Alibaba."
    
    final_prompt = wrapper.wrap_prompt(
        final_instruction,
        final_content,
        final_system
    )[0]
    prompt_tokens = client.count_tokens(final_prompt)["count"]
    if max_context > prompt_tokens + 500:
        max_generation = (max_context - prompt_tokens)
    else:
        print(f"Ran out of context for final summary, need {prompt_tokens}, have {max_context}")
        return 
    return client.generate(
        prompt=final_prompt,
        max_length=max_generation,
        min_p=0,
        rep_pen=1,
        temperature=0.1,
        top_k=0,
        top_p=1
    )

def main():
    parser = argparse.ArgumentParser(description="Analyze video using Qwen2vl 72B and KoboldCPP")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--api-url", default="http://localhost:5001",
                      help="KoboldCPP API URL (default: http://localhost:5001)")
    parser.add_argument("--templates", default="./templates",
                      help="Path to instruction templates (default: ./templates)")
    parser.add_argument("--max-frames", type=int, default=64,
                      help="Maximum frames to analyze (default: 64)")
    parser.add_argument("--output-dir",
                      help="Output directory (default: video_name_analysis)")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Frames per batch (default: 8, max 8)")                        
    args = parser.parse_args()
    
    try:
        results = analyze_video(
            args.video,
            args.api_url,
            args.templates,
            args.max_frames,
            args.output_dir,
            args.batch_size,
        )
        print("\nVideo Summary:")
        print("-" * 80)
        print(results["final_summary"])
        print("\nSaved to output directory.")
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())

        