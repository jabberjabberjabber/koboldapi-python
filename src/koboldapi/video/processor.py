"""Process videos through KoboldCPP API."""

import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Literal
import numpy as np
from PIL import Image

try:
    import decord
except ImportError:
    raise ImportError(
        "Video processing requires decord library. "
        "Install with: pip install decord"
    )

from ..core import KoboldAPICore
from ..utils.image_utils import (
    calculate_resize_dimensions,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS
)
from ..utils.qwen_resizer import (
    qwen_resize,
    qwen_frame_count,
    IMAGE_FACTOR,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FPS,
    FRAME_FACTOR
)

class VideoProcessor:
    """ Process videos through KoboldCPP API """
    
    def __init__(self, core: KoboldAPICore,
                 resize_mode: Literal['standard', 'qwen'] = 'standard',
                 max_pixels: Optional[int] = None,
                 min_pixels: Optional[int] = None):
        """ Initialize processor
        
            Args:
                core: KoboldAPICore instance
                resize_mode: Resizing algorithm to use ('standard' or 'qwen')
                max_pixels: Maximum pixels for standard resizing
                min_pixels: Minimum pixels for standard resizing
        """
        self.core = core
        self.resize_mode = resize_mode
        self.max_pixels = max_pixels or DEFAULT_MAX_PIXELS
        self.min_pixels = min_pixels or DEFAULT_MIN_PIXELS

    def _resize_frame(self, img: Image.Image, batch_size: int) -> Image.Image:
        """ Resize frame according to selected strategy.
        
            Args:
                img: PIL Image to resize
                batch_size: Number of frames in batch (for Qwen scaling)
                
            Returns:
                Resized PIL Image
        """
        width, height = img.size
        
        if self.resize_mode == 'qwen':
            # Adjust max pixels based on batch size for Qwen
            max_pixels = max(
                min(VIDEO_MAX_PIXELS, 
                    VIDEO_TOTAL_PIXELS / batch_size * FRAME_FACTOR),
                int(VIDEO_MIN_PIXELS * 1.05)
            )
            new_height, new_width = qwen_resize(
                height, width,
                factor=IMAGE_FACTOR,
                min_pixels=VIDEO_MIN_PIXELS,
                max_pixels=max_pixels
            )
        else:
            # For standard mode, divide max pixels by batch size
            batch_max = self.max_pixels // batch_size
            new_width, new_height = calculate_resize_dimensions(
                width, height,
                max_pixels=batch_max,
                min_pixels=self.min_pixels
            )
            
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    def process_video(self, video_path: Union[str, Path],
                     max_frames: int = 64) -> Tuple[List[str], float]:
        """ Process video and return encoded frames.
        
            Args:
                video_path: Path to video file
                max_frames: Maximum frames to extract
                
            Returns:
                Tuple of (base64 encoded frames, video length in seconds)
        """
        video_path = str(video_path)
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        
        # Determine number of frames to extract
        if self.resize_mode == 'qwen':
            n_frames = qwen_frame_count(total_frames, video_fps, max_frames)
        else:
            # Simple linear sampling for standard mode
            n_frames = min(max_frames, total_frames)
            
        video_length = total_frames / video_fps
        indices = np.linspace(0, total_frames - 1, n_frames).round().astype(int).tolist()
        
        # Ensure even number of frames
        if len(indices) % 2 != 0:
            indices.append(indices[-1])

        frames = vr.get_batch(indices).asnumpy()
        base64_frames = []
        
        for frame in frames:
            image = Image.fromarray(frame)
            image = self._resize_frame(image, len(indices))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            base64_frames.append(img_str)
        
        return base64_frames, video_length

    def analyze_batch(self, frame_batch: List[str], batch_idx: int,
                     batch_size: int, total_length: float) -> str:
        """ Analyze a batch of frames.
        
            Args:
                frame_batch: List of base64 encoded frames
                batch_idx: Index of current batch
                batch_size: Size of each batch
                total_length: Total video length in seconds
                
            Returns:
                Analysis text
        """
        instruction = (
            f"These are frames {' half ' if self.resize_mode == 'qwen' else ' '}"
            f"second apart from a video. "
            f"The total length of the video is {total_length:.1f}s. "
            f"Out of {batch_size} groups evenly divided in linear time, "
            f"this frame group is number {batch_idx}. "
            "Describe the action occurring over this time period. "
            "Pay close attention to characters, style, movement and scene. "
            "Make assumptions as needed using knowledge of common video "
            "content, themes, and tropes."
        )
        
        frame_system = (
            "You are a helpful assistant analyzing video content."
        )
        
        frame_prompt = self.core.template_wrapper.wrap_prompt(
            instruction,
            "",
            frame_system
        )[0]
        
        return self.core.api_client.generate(
            prompt=frame_prompt,
            images=frame_batch,
            max_length=500,
            min_p=0,
            rep_pen=1,
            temperature=0.1,
            top_k=0,
            top_p=1
        )

    def generate_summary(self, batch_analyses: List[Dict],
                        total_length: float) -> Optional[str]:
        """ Generate final summary from batch analyses.
        
            Args:
                batch_analyses: List of batch analysis results
                total_length: Total video length in seconds
                
            Returns:
                Summary text or None if context length exceeded
        """
        max_context = self.core.api_client.get_max_context_length()
        all_analyses = "\n\n".join(
            f"Group: {analysis['batch']}, Events: {analysis['analysis']}"
            for analysis in batch_analyses
        )
        
        final_instruction = (
            "Recall the events in a linear sequence to create a description "
            "of the entire video. Use as much of the individual descriptions "