#!/usr/bin/env python3
"""
Video analysis example using KoboldAPI's VideoProcessor.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from koboldapi import KoboldAPICore, VideoProcessor


def process_video(config: dict,
                 video_path: Path,
                 output_dir: Optional[Path] = None,
                 max_frames: int = 64,
                 batch_size: int = 8,
                 resize_mode: str = 'qwen') -> int:
    """ Process a video file and save results.
    
        Args:
            config: Configuration dictionary for KoboldAPICore
            video_path: Path to video file
            output_dir: Optional directory to save results
            max_frames: Maximum frames to analyze
            batch_size: Frames per batch
            resize_mode: Resizing algorithm to use ('standard' or 'qwen')
            
        Returns:
            Exit code (0 for success, 1 for errors)
    """
    try:
        core = KoboldAPICore(config)
        processor = VideoProcessor(
            core,
            resize_mode=resize_mode
        )
        
        results = processor.analyze_video(
            video_path,
            max_frames=max_frames,
            output_dir=output_dir,
            batch_size=batch_size
        )
        
        if results.get("final_summary"):
            print("\nVideo Summary:")
            print("-" * 80)
            print(results["final_summary"])
            print("\nFull analysis saved to output directory.")
        else:
            print("\nWarning: No final summary generated.")
            
        return 0
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return 1


def main():
    """ Main entry point with argument parsing. """
    parser = argparse.ArgumentParser(
        description="Analyze videos using KoboldAPI video processor"
    )
    
    parser.add_argument(
        "video",
        help="Path to video file"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5001",
        help="KoboldCPP API URL (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--templates",
        default="./templates",
        help="Path to instruction templates (default: ./templates)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Maximum frames to analyze (default: 64)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: video_name_analysis)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per batch (default: 8, max 8)"
    )
    parser.add_argument(
        "--resize-mode",
        choices=['standard', 'qwen'],
        default='qwen',
        help="Frame resizing algorithm (default: qwen)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    config_dict = {
        "api_url": args.api_url,
        "api_password": "",
        "templates_directory": args.templates,
        "min_p": 0,
        "rep_pen": 1,
        "temperature": 0.1,
        "top_k": 0,
        "top_p": 1
    }
    
    return process_video(
        config_dict,
        video_path,
        output_dir,
        args.max_frames,
        args.batch_size,
        args.resize_mode
    )


if __name__ == "__main__":
    exit(main())
