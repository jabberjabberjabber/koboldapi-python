"""Default instruction templates for various LLM models."""

from typing import Dict, List

# Template format for each model family
DEFAULT_TEMPLATES: Dict[str, Dict] = {
    "llama2": {
        "name": ["llama-2", "llama2", "llama 2"],
        "system_start": "<s>[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",
        "user_end": " [/INST]",
        "assistant_start": " ",
        "assistant_end": "</s>"
    },
    "mistral": {
        "name": ["mistral", "mixtral"],
        "system_start": "<s>[INST] ",
        "system_end": "\n",
        "user_start": "",
        "user_end": " [/INST]",
        "assistant_start": " ",
        "assistant_end": "</s>"
    },
    "openchat": {
        "name": ["openchat"],
        "system_start": "GPT4 Correct User: ",
        "system_end": "\n",
        "user_start": "Human: ",
        "user_end": "\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "<|end_of_turn|>"
    },
    "codellama": {
        "name": ["codellama", "code-llama", "code llama"],
        "system_start": "<s>[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",
        "user_end": " [/INST]",
        "assistant_start": " ",
        "assistant_end": "</s>"
    },
    "vicuna": {
        "name": ["vicuna"],
        "system_start": "SYSTEM: ",
        "system_end": "\n\n",
        "user_start": "USER: ",
        "user_end": "\n",
        "assistant_start": "ASSISTANT: ",
        "assistant_end": "\n"
    },
    "yi": {
        "name": ["yi"],
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n"
    },
    "qwen": {
        "name": ["qwen"],
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n"
    },
    "neural-chat": {
        "name": ["neural-chat", "neural chat"],
        "system_start": "### System:\n",
        "system_end": "\n\n",
        "user_start": "### User:\n",
        "user_end": "\n\n",
        "assistant_start": "### Assistant:\n",
        "assistant_end": "\n\n"
    },
    "default": {
        "name": ["default"],
        "system_start": "System: ",
        "system_end": "\n\n",
        "user_start": "User: ",
        "user_end": "\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "\n"
    }
}

def find_template(model_name: str) -> Dict:
    """Find the appropriate template for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Template dictionary for the model
    """
    model_name = model_name.lower()
    
    # Check each template's name list for a match
    for template in DEFAULT_TEMPLATES.values():
        if any(name in model_name for name in template["name"]):
            return template
            
    # Return default template if no match found
    return DEFAULT_TEMPLATES["default"]