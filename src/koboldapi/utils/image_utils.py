"""Generic utilities for image and video processing."""

from typing import Tuple, Optional
import math

# Default sizes for image processing
DEFAULT_MAX_PIXELS = 1920 * 1080  # 1080p
DEFAULT_MIN_PIXELS = 256 * 256    # Minimum readable size
DEFAULT_TOKEN_TARGET = 1000       # Target token count

def calculate_resize_dimensions(width: int, height: int, 
                              max_pixels: Optional[int] = None,
                              min_pixels: Optional[int] = None) -> Tuple[int, int]:
    """ Calculate new dimensions maintaining aspect ratio.
    
        Args:
            width: Original width
            height: Original height
            max_pixels: Maximum total pixels allowed
            min_pixels: Minimum total pixels required
            
        Returns:
            Tuple of (new_width, new_height)
    """
    max_pixels = max_pixels or DEFAULT_MAX_PIXELS
    min_pixels = min_pixels or DEFAULT_MIN_PIXELS
    
    current_pixels = width * height
    
    # Handle oversized images
    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Ensure even dimensions
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        return new_width, new_height
        
    # Handle undersized images
    if current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Ensure even dimensions
        new_width = new_width + (new_width % 2)
        new_height = new_height + (new_height % 2)
        return new_width, new_height
        
    # Image is within bounds
    return width, height

def estimate_image_tokens(width: int, height: int) -> int:
    """ Estimate number of tokens an image of given size might use.
    
        This is a rough estimation and may vary by model.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Estimated token count
    """
    # This is a simplified estimation - actual token count
    # will depend on the model and image complexity
    pixels = width * height
    return int(pixels / 2048)  # Rough approximation