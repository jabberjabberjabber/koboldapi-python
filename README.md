# KoboldCPP Integration Library

A Python library for interacting with KoboldCPP's API, allowing anyone to create basic scripts to utilize powerful LLMs locally for tasks using documents, images, and videos.

## Features

- Text processing with chunking
- Image and video processing
- Configurable API settings and generation parameters
- Wraps prompts with appropriate instruction templates for most models
- Support for custom instruction templates

## Installation

```bash
pip install koboldapi
```

or

```bash
git clone https://github.com/jabberjabberjabber/koboldapi-python/
pip install -e .
```

## Quick Start

### Basic Text Generation

```python
from koboldapi import KoboldAPICore

# Create core instance
core = KoboldAPICore()

# Generate text
response = core.api_client.generate(
    prompt="Tell me a story about a robot",
    max_length=300,
    temperature=0.7
)
print(response)
```

### Image Analysis

```python
from koboldapi import ImageProcessor

# Initialize image processor
image_processor = ImageProcessor(core)

# Analyze an image
result, output_path = image_processor.process_image(
    "path/to/image.jpg",
    instruction="Describe what you see in this image",
    temperature=0.1
)
print(result)
```

### Video Processing

```python
from koboldapi import VideoProcessor

# Initialize video processor
video_processor = VideoProcessor(core)

# Analyze a video
results = video_processor.analyze_video(
    "path/to/video.mp4"
)
print(results["final_summary"])
```

## Core Components

### KoboldAPICore

The central component that manages API communication and provides access to core functionality.

```python
from koboldapi import KoboldAPICore

core = KoboldAPICore()

# Get model information
model_info = core.get_model_info()

# Test connection
is_connected = core.validate_connection()
```

### Configuration

Use `KoboldAPIConfig` to manage API settings and generation parameters:

```python
from koboldapi import KoboldAPIConfig

config = KoboldAPIConfig(
    api_url="http://localhost:5001",
    api_password="your_password",
    templates_directory="./templates",
    translation_language="English",
    text_completion=False,
    temp=0.2,
    top_k=0,
    top_p=1.0,
    rep_pen=1.1,
    min_p=0.02
)

# Save configuration
config.to_json("config.json")

# Load configuration
loaded_config = KoboldAPIConfig.from_json("config.json")
```

## Text Processing

### Chunking

The `ChunkingProcessor` splits large texts into manageable chunks for LLM processing:

```python
from koboldapi import ChunkingProcessor

chunker = ChunkingProcessor(core.api_client, max_chunk_length=2048)

# Process text
chunks = chunker.chunk_text("Your long text content here")

# Process file (supports many document types)
chunks, metadata = chunker.chunk_file("path/to/document.pdf")
```

### Templates

The library supports will wrap content with a detected template or a custom template:

```python
from koboldapi import InstructTemplate

template_wrapper = InstructTemplate(
    url="http://localhost:5001"
)

# Format prompt with template
formatted_prompt = template_wrapper.wrap_prompt(
    instruction="Analyze this text",
    content="Content to analyze",
    system_instruction="You are a helpful assistant"
)
```

## Image Processing

The `ImageProcessor` handles image analysis with automatic resizing and optimization:

```python
from koboldapi import ImageProcessor

processor = ImageProcessor(
    core,
    resize_mode='standard',  # or 'qwen'
    max_pixels=1920 * 1080,
    min_pixels=256 * 256
)

# Process single image
result, output_path = processor.process_image(
    "image.jpg",
    instruction="Describe this image",
    temperature=0.1
)

# Process batch of images
results = processor.process_batch(
    ["image1.jpg", "image2.jpg"],
    instruction="Describe each image",
    output_dir="./output"
)
```

## Video Processing

The `VideoProcessor` handles video analysis with frame extraction and sequential processing:

```python
from koboldapi import VideoProcessor

processor = VideoProcessor(
    core,
    resize_mode='standard',  # or 'qwen'
    max_pixels=1920 * 1080,
    min_pixels=256 * 256
)

# Analyze video
results = processor.analyze_video(
    "video.mp4",
    max_frames=64,
    batch_size=8,
    output_dir="./output"
)

# Access results
print(results["final_summary"])
for analysis in results["analysis"]:
    print(f"Batch {analysis['batch']}: {analysis['analysis']}")
```

## Utilities

### Image Utilities

```python
from koboldapi.utils.image_utils import (
    calculate_resize_dimensions,
    estimate_image_tokens
)

# Calculate new dimensions
new_width, new_height = calculate_resize_dimensions(
    width=1920,
    height=1080,
    max_pixels=1920 * 1080,
    min_pixels=256 * 256
)

# Estimate tokens
token_count = estimate_image_tokens(width=800, height=600)
```

### Qwen Resizer

For Qwen-VL compatible resizing:

```python
from koboldapi.utils.qwen_resizer import (
    qwen_resize,
    qwen_frame_count
)

# Resize dimensions
new_height, new_width = qwen_resize(
    height=1080,
    width=1920,
    factor=28
)

# Calculate frame count
n_frames = qwen_frame_count(
    total_frames=100,
    video_fps=30,
    fps_max_frames=64
)
```

## Error Handling

The library provides custom exceptions for error handling:

```python
from koboldapi import KoboldAPIError

try:
    result = core.api_client.generate("prompt")
except KoboldAPIError as e:
    print(f"API error: {e}")
```

## Configuration Files

Example JSON configuration file:

```json
{
    "api_url": "http://localhost:5001",
    "api_password": "your_password",
    "templates_directory": "./templates",
    "translation_language": "English",
    "text_completion": false,
    "temp": 0.2,
    "top_k": 0,
    "top_p": 1.0,
    "rep_pen": 1.1,
    "min_p": 0.02
}
```

## Practical Examples

Here are some practical examples demonstrating real-world applications of the library.

### Package Theft Detection from Security Footage

```python
from koboldapi import KoboldAPICore, VideoProcessor

# Initialize
core = KoboldAPICore({
    "api_url": "http://localhost:5001",
    "api_password": "",
    "templates_directory": "./templates"
})

# Create security-focused video processor
processor = VideoProcessor(
    core,
    resize_mode='qwen'  # Better for detail detection
)

# Configure security-specific prompts
system_prompt = """You are a security analysis assistant. Focus on identifying 
and describing any suspicious behavior, particularly related to packages or 
deliveries. Note timestamps, descriptions of individuals, and potential 
security concerns."""

instruction = """Analyze this segment of security footage. Look specifically for:
1. Any interaction with packages or mail
2. People approaching the property
3. Suspicious behavior
4. Vehicle descriptions
Provide timestamps and detailed descriptions of relevant activity."""

# Process video with security focus
results = processor.analyze_video(
    "security_footage.mp4",
    max_frames=128,  # More frames for better coverage
    batch_size=8,
    output_dir="security_analysis",
    system_instruction=system_prompt,
    instruction=instruction
)

# Print analysis
print(results["final_summary"])
```

### OCR for Handwritten Text

```python
from koboldapi import KoboldAPICore, ImageProcessor

# Initialize
core = KoboldAPICore({
    "api_url": "http://localhost:5001",
    "api_password": "", # optional
    "templates_directory": "./templates" # optional
})

processor = ImageProcessor(core)

system_prompt = """You are an OCR assistant specializing in handwriting recognition.
Carefully analyze the handwriting and provide:
1. The exact transcribed text
2. Confidence notes about any unclear words
3. Alternative interpretations for ambiguous writing"""

instruction = """Transcribe the handwritten text in this image. If any parts are 
unclear, provide your best interpretation and note your uncertainty. Maintain 
original line breaks and formatting where visible."""

processor.process_batch(
    ["handwriting1.png", "handwriting2.jpeg"],
    instruction=instruction,
    system_instruction=system_prompt,
    output_dir="ocr_results", # Will save results here
    temperature=0.1  # Low temperature required
)
```

### Document Formatting and Professionalization

```python
from pathlib import Path
from koboldapi import KoboldAPICore, ChunkingProcessor

# Initialize
core = KoboldAPICore({
    "api_url": "http://localhost:5001",
    "api_password": "",
    "templates_directory": "./templates"
})

# Create processor for chunking documents
processor = ChunkingProcessor(
    core.api_client,
    max_chunk_length=2048
)

# Get document chunks
doc_path = Path("draft_document.txt")  # Can also be PDF
chunks, metadata = processor.chunk_file(doc_path)

# Configure formatting prompts
system_prompt = """You are a professional document editor. Improve the text while 
maintaining its core meaning and intent. Focus on:
1. Correct spelling and grammar
2. Professional tone and vocabulary
3. Consistent formatting
4. Clear paragraph structure"""

instruction = """Edit this text to be more professional while keeping its meaning.
Fix any errors in spelling, grammar, or formatting. Ensure consistency in style 
and maintain appropriate business tone."""

# Process each chunk
formatted_chunks = []
for chunk, _ in chunks:
    prompt = core.template_wrapper.wrap_prompt(
        instruction=instruction,
        content=chunk,
        system_instruction=system_prompt
    )
    
    result = core.api_client.generate(
        prompt=prompt,
        max_length=len(chunk) // 2,
        temperature=0.1,
        top_p=0.9,
        rep_pen=1.1
    )
    formatted_chunks.append(result)

# Save formatted document
output_path = Path("formatted_document.txt")
output_path.write_text(
    "\n\n".join(formatted_chunks),
    encoding='utf-8'
)
```

These examples demonstrate how to use the library for specific real-world tasks. Each example:
- Uses appropriate processor types
- Configures task-specific prompts
- Adjusts parameters for optimal results
- Includes proper error handling
- Provides clear output formatting

The library's modular design makes it easy to adapt these examples for similar use cases or combine them for more complex applications.

## Template Files

If you have a model with a custom instruct template, you can create an adapter for it.

Example instruction template (templates/alpaca.json):

```json
{
    "name": ["alpaca"],
    "system_start": "### System:\n",
    "system_end": "\n\n",
    "user_start": "### Human: ",
    "user_end": "\n\n",
    "assistant_start": "### Assistant: "
}
```
