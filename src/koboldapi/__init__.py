"""
KoboldAPI - Python library for interacting with KoboldCPP API

This library provides high-level interfaces for:
- Text generation and chat
- Content processing (text, images, video)
- Template management
- Configuration handling
"""

from .core import (
    KoboldAPI,
    KoboldAPIError,
    KoboldAPICore,
    InstructTemplate
)
from .image import ImageProcessor
from .chunking import ChunkingProcessor

__version__ = "0.5.0"
__author__ = "jabberjabberjabber"
__license__ = "GNU General Public License v3.0"

__all__ = [
    # Core components
    'KoboldAPI',
    'KoboldAPIError',
    'KoboldAPICore',
    'InstructTemplate',
    
    # Processors
    'ImageProcessor',
    'ChunkingProcessor',
]