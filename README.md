# Basic functions

## KoboldCpp API Interface

Allows easy use of basic KoboldCpp API endpoints, including streaming generations, images, samplers.

## Instruct Template Wrapping

Finds the appropriate instruct template for the running model and wraps it around content to create a prompt.
 
## Chunking

Will read most types of document and chunk them any size up to max context. Stops at natural break points. Returns the chunks as a list.

# Guide to Using the KoboldCPP API with Python

## Introduction

KoboldCPP is a powerful and portable solution for running Large Language Models (LLMs). Its standout features include:

- Zero-installation deployment with single executable
- Support for any GGUF model compatible with LlamaCPP
- Cross-platform support (Linux, Windows, macOS)
- Hardware acceleration via CUDA and Vulkan
- Built-in GUI with extensive features
- Multimodal capabilities (image generation, speech, etc.)
- API compatibility with OpenAI and Ollama

## Quick Start

### Basic Setup

1. Download the KoboldCPP executable for your platform
2. Place your GGUF model file in the same directory
3. Install the Python client:

```bash
git clone https://github.com/jabberjabberjabber/koboldapi-python
cd koboldapi-python
pip install git+https://github.com/jabberjabberjabber/koboldapi-python.git
```

### First Steps

Here's a minimal example to get started:

```python
from koboldapi import KoboldAPI

# Initialize the client
api = KoboldAPI("http://localhost:5001")

# Basic text generation
response = api.generate(
    prompt="Write a haiku about programming:",
    max_length=50,
    temperature=0.7
)
print(response)
```

## Core Concepts

### Configuration Management

The `KoboldAPIConfig` class manages configuration settings for the API client. You can either create a config programmatically or load it from a JSON file:

```python
from koboldapi import KoboldAPIConfig

# Create config programmatically
config = KoboldAPIConfig(
    api_url="http://localhost:5001",
    api_password="",
    templates_directory="./templates",
    translation_language="English",
    temp=0.7,
    top_k=40,
    top_p=0.9,
    rep_pen=1.1
)

# Or load from JSON file
config = KoboldAPIConfig.from_json("config.json")

# Save config to file
config.to_json("new_config.json")
```

Example config.json:
```json
{
    "api_url": "http://localhost:5001",
    "api_password": "",
    "templates_directory": "./templates",
    "translation_language": "English",
    "temp": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "rep_pen": 1.1
}
```

### Template Management

KoboldAPI supports various instruction formats through templates. The `InstructTemplate` class handles this automatically:

```python
from koboldapi.templates import InstructTemplate

template = InstructTemplate("./templates", "http://localhost:5001")

# Wrap a prompt with the appropriate template
wrapped_prompt = template.wrap_prompt(
    instruction="Explain quantum computing",
    content="Focus on qubits and superposition",
    system_instruction="You are a quantum physics expert"
)
```

## Example Applications

### Text Processing

The library includes example scripts for various text processing tasks:

```python
from koboldapi import KoboldAPICore
from koboldapi.chunking.processor import ChunkingProcessor

# Initialize core with config
config = {
    "api_url": "http://localhost:5001",
    "templates_directory": "./templates"
}
core = KoboldAPICore(config)

# Process a text file
processor = ChunkingProcessor(core.api_client, max_chunk_length=2048)
chunks, metadata = processor.chunk_file("document.txt")

# Generate summary for each chunk
for chunk, _ in chunks:
    summary = core.api_client.generate(
        prompt=core.template_wrapper.wrap_prompt(
            instruction="Summarize this text",
            content=chunk
        )[0],
        max_length=200
    )
    print(summary)
```

### Image Processing

Process images:

```python
from koboldapi import KoboldAPICore
from pathlib import Path

# Initialize core
config = {
    "api_url": "http://localhost:5001",
    "templates_directory": "./templates"
}
core = KoboldAPICore(config)

# Process image
image_path = Path("image.png")
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

result = core.api_client.generate(
    prompt=core.template_wrapper.wrap_prompt(
        instruction="Extract text from this image",
        system_instruction="You are an OCR system"
    )[0],
    images=[image_data],
    temperature=0.1
)
print(result)
```


## Advanced Features

### Custom Template Creation

Create custom instruction templates for different models:

```python
{
    "name": ["vicuna-7b", "vicuna-13b"],
    "system_start": "### System:\n",
    "system_end": "\n\n",
    "user_start": "### Human: ",
    "user_end": "\n\n",
    "assistant_start": "### Assistant: ",
    "assistant_end": "\n\n"
}
```

### Generation Parameters

Fine-tune generation settings:

```python
response = api.generate(
    prompt="Write a story:",
    max_length=500,
    temperature=0.8,      # Higher = more creative
    top_p=0.9,           # Nucleus sampling threshold
    top_k=40,            # Top-k sampling threshold
    rep_pen=1.1,         # Repetition penalty
    rep_pen_range=256,   # How far back to apply rep penalty
    min_p=0.05          # Minimum probability threshold
)
```

### Error Handling

Implement robust error handling:

```python
from koboldapi import KoboldAPIError

try:
    response = api.generate(prompt="Test prompt")
except KoboldAPIError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Context Management

Optimize token usage:

```python
# Get max context length
max_length = api.get_max_context_length()

# Count tokens in prompt
token_count = api.count_tokens(prompt)["count"]

# Ensure we stay within limits
available_tokens = max_length - token_count
response_length = min(desired_length, available_tokens)
```

### Batch Processing

Handle multiple inputs efficiently:

```python
async def process_batch(prompts):
    results = []
    for prompt in prompts:
        async for token in api.stream_generate(prompt):
            results.append(token)
    return results
```

## Troubleshooting

### Common Issues

1. Connection Errors
```python
# Test connection
if not api.validate_connection():
    print("Cannot connect to API")
```

2. Template Errors
```python
# Check if template exists
if not template.get_template():
    print("No matching template found for model")
```

3. Generation Errors
```python
# Monitor generation status
status = api.check_generation()
if status is None:
    print("Generation failed or was interrupted")
```

## Contributing

Contributions to improve these tools are welcome. Please submit issues and pull requests on GitHub.

### Development Setup

1. Clone the repository
2. Install development dependencies:
```bash
pip install -e ".[dev]"
```
3. Run tests:
```bash
pytest tests/
```

## License

This project is licensed under the GPLv3 license.