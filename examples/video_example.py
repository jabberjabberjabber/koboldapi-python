import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import glob
import os
from typing import List, Dict

from koboldapi import KoboldAPI, InstructTemplate, LLMConfig
from koboldapi.core.core import LLMToolsCore

# recommended models:
# minicpm-v-2.6, qvq

# requires ffmpeg installed and on path

def get_scene_frames(video_path: str, threshold: float = 0.1, 
                    min_gap: int = 4, format: str = 'png') -> List[str]:
    """Gets scene change frames with frame numbers and timecode.
    
    Args:
        video_path: Path to input video
        threshold: Scene detection threshold (0.0-1.0) 
        min_gap: Minimum seconds between detected scenes
        format: Output image format ('jpeg' or 'png')
        
    Returns:
        List of Base64 encoded images for detected scenes
    """
    resources_dir = Path(__file__).parent.parent / 'resources'
    font_path = resources_dir / 'arial.ttf'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_pattern = str(tmpdir / f'scene_%04d.{format}')
        
        filter_expr = (
            f"select='if(gt(scene,{threshold}),scene,0)"
            f"+if(isnan(prev_selected_t),0,gte(t-prev_selected_t,{min_gap}))"
            f"*between(t-prev_selected_t,-{min_gap},{min_gap})',"
            "setpts=N/FRAME_RATE/TB,"
            "scale='if(gt(iw,ih),min(320,iw),-1):if(gt(iw,ih),-1,min(320,ih))',"
            f"drawtext=fontfile='{font_path}'"
            ":text='FRAME\\:%{n} TIME\\:%{pts\\:hms} TIMECODE\\:%{pts\\:hms\\:24}'"
            ":x=10:y=10:fontcolor=white:fontsize=24:box=1:boxcolor=black"
        )
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', filter_expr,
            '-vsync', 'vfr',
            output_pattern
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        return [
            base64.b64encode(frame_path.read_bytes()).decode()
            for frame_path in sorted(tmpdir.glob(f'scene_*.{format}'))
        ]

def analyze_video(video_path: str, config: LLMConfig,
                 max_frames: int = 84, output_dir: str = None,
                 batch_size: int = 4) -> Dict:
    """Analyze video using KoboldCPP API.

    Args:
        video_path: Path to video file
        config: LLMConfig instance
        max_frames: Maximum number of frames to process
        output_dir: Directory to save results (defaults to video location)
        batch_size: Number of frames to process in each batch

    Returns:
        Dict containing analysis results and metadata
    """
    core = LLMToolsCore(config)
    client = core.api
    wrapper = core.template
    
    video_path = Path(video_path)
    out_path = Path(output_dir) if output_dir else video_path.parent / f"{video_path.stem}_analysis"
    out_path.mkdir(exist_ok=True)
    
    results = {
        "analysis": [],
        "progressive_summaries": [],
        "final_summary": None,
        "metadata": {
            "video_path": str(video_path.absolute()),
            "api_url": config.api_url,
            "max_frames": max_frames,
            "batch_size": batch_size
        }
    }
    
    print(f"Extracting frames from {video_path}...")
    frames = get_scene_frames(video_path)[:max_frames]
    results["metadata"]["frame_count"] = len(frames)
    
    frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    
    for batch_idx, frame_batch in enumerate(frame_batches, 1):
        print(f"Processing batch {batch_idx}/{len(frame_batches)}...")
        
        # Analyze frame batch
        analysis = client.generate(
            prompt=wrapper.wrap_prompt(
                "Describe the objects and actions in these frames.",
                system_instruction="You are an expert video analyzer."
            )[0],
            images=frame_batch,
            temperature=0.1,
            max_length=200
        )
        
        results["analysis"].append({
            "batch": batch_idx,
            "frame_range": f"{(batch_idx-1)*batch_size + 1}-{min(batch_idx*batch_size, len(frames))}",
            "analysis": analysis
        })
    
    if results["analysis"]:
        # Generate final summary
        all_analyses = "\n\n".join(
            f"Frames {a['frame_range']}: {a['analysis']}" 
            for a in results["analysis"]
        )
        
        results["final_summary"] = client.generate(
            prompt=wrapper.wrap_prompt(
                "Summarize these video frames into a coherent narrative.",
                content=f"Frame analyses:\n{all_analyses}",
                system_instruction="You are an expert video analyzer. Use plain language."
            )[0],
            temperature=0,
            max_length=500
        )
    
    # Save results
    with open(out_path / "analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze video using KoboldCPP")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--api-url", default="http://localhost:5001",
                      help="KoboldCPP API URL (default: http://localhost:5001)")
    parser.add_argument("--template-dir", default="./templates",
                      help="Path to instruction templates (default: ./templates)")
    parser.add_argument("--max-frames", type=int, default=24,
                      help="Maximum frames to analyze (default: 24)")
    parser.add_argument("--output-dir",
                      help="Output directory (default: video_name_analysis)")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Frames per batch (default: 1)")
    
    args = parser.parse_args()
    
    try:
        # Set up configuration
        if args.config:
            config = LLMConfig.from_json(args.config)
        else:
            config = LLMConfig(
                api_url=args.api_url,
                api_password="",
                templates_directory=args.template_dir
            )
        
        # Process video
        results = analyze_video(
            args.video,
            config,
            args.max_frames,
            args.output_dir,
            args.batch_size
        )
        
        # Display results
        print("\nVideo Analysis Summary:")
        print("-" * 80)
        print(results["final_summary"])
        print(f"\nFull analysis saved to: {args.output_dir or Path(args.video).parent}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())