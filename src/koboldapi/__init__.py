"""
KoboldAPI - Python library for interacting with KoboldCPP API

This library provides high-level interfaces for:
- Text generation and chat  
- Content processing (text, images, video)
- Template management
- Configuration handling
"""

from .core.api import KoboldAPI, KoboldAPIError
from .core.core import KoboldAPICore  
from .core.templates import InstructTemplate
from .image.processor import ImageProcessor
from .chunking.processor import ChunkingProcessor

__version__ = "0.5.2"
__author__ = "jabberjabberjabber"
__license__ = "GNU General Public License v3.0"

__all__ = [
    'KoboldAPI',
    'KoboldAPIError', 
    'KoboldAPICore',
    'InstructTemplate',
    'ImageProcessor',
    'ChunkingProcessor',
]