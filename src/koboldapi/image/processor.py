import base64
import io
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Literal
import sys

from PIL import Image
from ..core import KoboldAPICore
from ..utils.image_utils import (
    calculate_resize_dimensions,
    estimate_image_tokens,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS
)
from ..utils.qwen_resizer import qwen_resize, IMAGE_FACTOR

class ImageProcessor:
    """ Process images through KoboldCPP API """
    
    def __init__(self, core: KoboldAPICore,
                 resize_mode: Literal['standard', 'qwen'] = 'standard',
                 max_pixels: Optional[int] = None,
                 min_pixels: Optional[int] = None):
        """ Initialize with KoboldAPICore instance
        
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

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """ Resize image according to selected strategy.
        
            Args:
                img: PIL Image to resize
                
            Returns:
                Resized PIL Image
        """
        width, height = img.size
        
        if self.resize_mode == 'qwen':
            new_height, new_width = qwen_resize(
                height, width,
                factor=IMAGE_FACTOR
            )
        else:  # standard
            new_width, new_height = calculate_resize_dimensions(
                width, height,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels
            )
            
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    def process_image(self, image_path: Union[str, Path],
                     instruction: str,
                     system_instruction: str = "You are a helpful assistant.",
                     temperature: float = 0.1,
                     top_p: float = 1.0,
                     top_k: int = 0,
                     rep_pen: float = 1.05) -> Tuple[Optional[str], Path]:
        """ Process a single image through the LLM.
        
            Args:
                image_path: Path to image file
                instruction: Instruction for processing the image
                system_instruction: System prompt for the model
                temperature: Sampling temperature
                top_p: Top-p sampling threshold
                top_k: Top-k sampling threshold
                rep_pen: Repetition penalty
                
            Returns:
                Tuple of (generated text or None if failed, output path)
        """
        image_path = Path(image_path)
        if not image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print(f"Unsupported file type: {image_path}", file=sys.stderr)
            return None, image_path
            
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = self._resize_image(img)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                encoded = base64.b64encode(buffer.getvalue()).decode()
                
                # Estimate tokens for logging
                width, height = img.size
                estimated_tokens = estimate_image_tokens(width, height)
                print(f"Estimated image tokens: {estimated_tokens}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            return None, image_path
            
        max_context = self.core.api_client.get_max_context_length()
        prompt_tokens = max_context // 4  # Reserve some for the image
        
        try:
            prompt = self.core.template_wrapper.wrap_prompt(
                instruction=instruction,
                system_instruction=system_instruction
            )
            
            result = self.core.api_client.generate(
                prompt=prompt,
                images=[encoded],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                rep_pen=rep_pen,
                max_length=prompt_tokens
            )
            return result, image_path.with_suffix('.txt')
        except Exception as e:
            print(f"Error generating response: {e}", file=sys.stderr)
            return None, image_path
            
    def process_batch(self, image_paths: List[Union[str, Path]],
                     instruction: str,
                     system_instruction: str = "You are a helpful assistant.",
                     output_dir: Optional[Union[str, Path]] = None,
                     **kwargs) -> Dict[str, Dict]:
        """ Process multiple images through the LLM.
        
            Args:
                image_paths: List of paths to image files
                instruction: Instruction for processing the images
                system_instruction: System prompt for the model
                output_dir: Optional directory to save results
                **kwargs: Additional parameters passed to process_image
                
            Returns:
                Dictionary mapping image paths to results and metadata
        """
        results = {}
        had_error = False
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(image_paths, 1):
            image_path = Path(image_path)
            print(f"Processing {image_path.name} ({i}/{len(image_paths)})...")
            
            result, default_output = self.process_image(
                image_path,
                instruction,
                system_instruction,
                **kwargs
            )
            
            # Determine output path
            if output_dir:
                output_path = output_dir / default_output.name
            else:
                output_path = default_output
                
            if result:
                try:
                    output_path.write_text(result, encoding='utf-8')
                    print(f"Saved output to {output_path}")
                    results[str(image_path)] = {
                        "success": True,
                        "output_path": str(output_path),
                        "result": result
                    }
                except Exception as e:
                    print(f"Error saving to {output_path}: {e}", file=sys.stderr)
                    had_error = True
                    results[str(image_path)] = {
                        "success": False,
                        "error": f"Failed to save output: {str(e)}"
                    }
            else:
                had_error = True
                results[str(image_path)] = {
                    "success": False,
                    "error": "Processing failed"
                }
                
        results["metadata"] = {
            "total_images": len(image_paths),
            "had_errors": had_error,
            "instruction": instruction,
            "system_instruction": system_instruction,
            "resize_mode": self.resize_mode,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels
        }
        
        return results