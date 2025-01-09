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

The `LLMConfig` class provides a clean way to manage your API settings:

```python
from koboldapi.config import LLMConfig

config = LLMConfig(
    api_url="http://localhost:5001",
    api_password=None,  # If you've set an API password
    templates_directory="./templates",
    temperature=0.7,
    top_p=0.9
)

# Save configuration
config.to_json("config.json")

# Load existing configuration
config = LLMConfig.from_json("config.json")
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

## Common Tasks

### Text Generation

```python
# Basic generation
text = api.generate(
    prompt="Write a short story:",
    max_length=200,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    rep_pen=1.1
)

# Streaming generation
async for token in api.stream_generate(
    prompt="Write a poem:",
    max_length=100
):
    print(token, end="", flush=True)
```

### Token Management

```python
# Count tokens in text
result = api.count_tokens("Hello, world!")
print(f"Token count: {result['count']}")
print(f"Token IDs: {result['tokens']}")

# Convert text to tokens
tokens = api.tokenize("Hello, world!")

# Convert tokens back to text
text = api.detokenize(tokens)
```

### Generation Control

```python
# Start generation
api.generate(
    prompt="Write a long story:",
    max_length=1000
)

# Check generation status
status = api.check_generation()

# Abort if needed
if api.abort_generation():
    print("Generation aborted successfully")
```

## Advanced Usage

### Custom Stop Sequences

```python
response = api.generate(
    prompt="List some colors:",
    stop_sequences=[".", "\n\n"],
    max_length=100
)
```

### Logging Probabilities

```python
response = api.generate(
    prompt="Predict the next word:",
    logprobs=True
)
logprobs = api.get_last_logprobs()
```

### Performance Monitoring

```python
# Get performance stats
stats = api.get_performance_stats()
print(f"Generation speed: {stats['tokens_per_second']} tokens/sec")

# Get model info
model_name = api.get_model()
max_context = api.get_max_context_length()
```


## Common Issues and Solutions

1. Connection Errors
   - Verify KoboldCPP is running and port is correct
   - Check for firewall restrictions
   - Ensure API password is correctly set if used

2. Performance Issues
   - Monitor GPU memory usage
   - Adjust batch size and context length
   - Consider using streaming for long generations

3. Template Mismatches
   - Verify template compatibility with model
   - Check template format and required fields
   - Use appropriate system instructions


## Contributing

Contributions to improve these tools are welcome. Please submit issues and pull requests on GitHub.

  