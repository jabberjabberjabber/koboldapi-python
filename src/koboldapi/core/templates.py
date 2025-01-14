"""Template system for instruction formatting."""

from jinja2.sandbox import ImmutableSandboxedEnvironment
from pathlib import Path
import json
import re
from typing import Optional, Dict, List, Union
import requests

from .api import KoboldAPI


DEFAULT_TEMPLATES: Dict[str, Dict] = {
    {
      "name": ["Alpaca"],
      "system_start": "",
      "system_end": "\n\n",
      "user_start": "### Instruction:\n",
      "user_end": "\n\n",
      "assistant_start": "### Response:\n",
      "assistant_end": "</s>\n\n"
    },
    {
      "name": ["ChatML", "obsidian", "Nous", "Hermes", "qwen", "MiniCPM-V-2.6", "QvQ", "QwQ"], 
      "system_start": "<|im_start|>system\n",
      "system_instruction": "You are a helpful assistant.",
      "system_end": "<|im_end|>\n",
      "user_start": "<|im_start|>user\n",
      "user_end": "<|im_end|>\n",
      "assistant_start": "<|im_start|>assistant\n",
      "assistant_end": "<|im_end|>\n"
    },
      "name": ["Command-r", "aya", "cmdr", "c4ai"],
      "system_start": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
      "system_end": "<|END_OF_TURN_TOKEN|>",
      "user_start": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
      "user_end": "<|END_OF_TURN_TOKEN|>",
      "assistant_start": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
      "assistant_end": "<|END_OF_TURN_TOKEN|>"
    },
    {
      "name": ["gemma-2", "gemma"],
      "system_start": "<start_of_turn>system\n",
      "system_end": "<end_of_turn>\n",
      "user_start": "<start_of_turn>user\n",
      "user_end": "<end_of_turn>\n",
      "assistant_start": "<start_of_turn>model\n",
      "assistant_end": "<end_of_turn>\n"
    },
    {
      "name": ["Llama-2"],
      "system_start": "",
      "system_end": "",
      "user_start": "<s>[INST] ",
      "user_end": "",
      "assistant_start": " [/INST] ",
      "assistant_end": " </s>\n"
    },
    {
      "name": ["Llama-3", "MiniCPM-V-2.5"],
      "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
      "system_end": "<|eot_id|>",
      "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
      "user_end": "<|eot_id|>",
      "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
      "assistant_end": "<|eot_id|>"
    },
    {
      "name": ["Metharme"],
      "system_start": "<|system|>",
      "system_end": "",
      "user_start": "<|user|>",
      "user_end": "",
      "assistant_start": "<|model>",
      "assistant_end": ""
    },
    {
      "name": ["Mistral", "Miqu", "Mixtral"], 
      "system_start": "",
      "system_end": "",
      "user_start": " [INST] ",
      "user_end": "",
      "assistant_start": " [/INST]",
      "assistant_end": "</s>"
    },
    {
      "name": ["Mistral Large", "Mistral Small", "Mistral 2409"], 
      "system_start": "",
      "system_end": "",
      "user_start": "[INST] ",
      "user_end": "",
      "assistant_start": "[/INST]",
      "assistant_end": "</s>"
    },
    {
      "name": ["Mistral Nemo"],
      "system_start": "",
      "system_end": "",
      "user_start": "[INST]",
      "user_end": "",
      "assistant_start": "[/INST]",
      "assistant_end": "</s>"
    },
    {
      "name": ["Phi"],
      "system_start": "<|system|>\n",
      "system_end": "<|end|>\n",
      "user_start": "<|user|>\n",
      "user_end": "<|end|>\n",
      "assistant_start": "<|assistant|>\n",
      "assistant_end": "<|end|>\n"
    },
    {
      "name": ["Vicuna", "Wizard"],
      "system_start": "",
      "system_end": "\n\n",
      "user_start": "USER: ",
      "user_end": "\n",
      "assistant_start": "ASSISTANT: ",
      "assistant_end": "</s>\n"
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
    
class InstructTemplate:
    """ Wraps instructions and content with appropriate templates. """
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None,
                 url: str = "http://localhost:5001"):
        """ Initialize template system.
        
            Args:
                templates_dir: Optional directory containing custom templates
                url: URL of KoboldCPP API
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.api_client = KoboldAPI(url)
        self.url = url
        self.model = self.api_client.get_model()
        self.jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def _normalize(self, s: str) -> str:
        """ Normalize string for comparison. """
        return re.sub(r"[^a-z0-9]", "", s.lower())
        
    def _get_adapter_template(self) -> Optional[Dict]:
        """ Get template from file or defaults.
        
            First checks custom templates directory if provided,
            then falls back to built-in defaults.
            
            Returns:
                Template dictionary or None if no match found
        """
        model_name_normalized = self._normalize(self.model)
        best_match = None
        best_match_length = 0
        best_match_version = 0
        
        # Check custom templates if directory provided
        if self.templates_dir and self.templates_dir.exists():
            try:
                for file in self.templates_dir.glob('*.json'):
                    with open(file) as f:
                        template = json.load(f)
                    required_fields = [
                        "name",
                        "system_start", "system_end",
                        "user_start", "user_end",
                        "assistant_start", "assistant_end"
                    ]
                    if not all(field in template for field in required_fields):
                        print(f"Template {file} missing required fields, skipping")
                        continue
            except Exception as e:
                print(f"Error reading template files: {str(e)}")

                    
        else:
            template = DEFAULT_TEMPLATES
            
        for name in template["name"]:
            normalized_name = self._normalize(name)
            if normalized_name in model_name_normalized:
                version_match = re.search(r'(\d+)(?:\.(\d+))?', name)
                current_version = float(f"{version_match.group(1)}.{version_match.group(2) or '0'}") if version_match else 0
                name_length = len(normalized_name)
                if current_version > best_match_version or \
                   (current_version == best_match_version and 
                    name_length > best_match_length):
                    best_match = template
                    best_match_length = name_length
                    best_match_version = current_version

        # If no custom template found, use built-in defaults
        if not best_match:
            best_match = DEFAULT_TEMPLATES.get("default")
            
        return best_match
        
    def _get_props(self) -> Optional[Dict]:
        """ Get template from props endpoint. """
        try:
            if not self.url.endswith('/props'):
                props_url = self.url.rstrip('/') + '/props'
            response = requests.get(props_url)
            response.raise_for_status()
            return response.json().get("chat_template")
        except: 
            return None

    def get_template(self) -> Dict:
        """ Get templates for the current model.
        
            Returns:
                Dictionary containing adapter and jinja templates
        """
        templates = {
            "adapter": self._get_adapter_template(),
            "jinja": self._get_props()
        }
        return templates
        
    def wrap_prompt(self, instruction: str, content: str = "",
                   system_instruction: str = "") -> List[str]:
        """ Format a prompt using templates.
        
            Args:
                instruction: Main instruction for the model
                content: Optional content to process
                system_instruction: Optional system-level instruction
                
            Returns:
                List of wrapped prompts (adapter and jinja versions)
        """
        user_text = f"{content}\n\n{instruction}" if content else instruction
        prompt_parts = []
        wrapped = []
        
        if adapter := self.get_template()["adapter"]:
            if system_instruction:
                prompt_parts.extend([
                    adapter["system_start"],
                    system_instruction,
                    adapter["system_end"]
                ])
            prompt_parts.extend([
                adapter["user_start"],
                user_text,
                adapter["user_end"],
                adapter["assistant_start"]
            ])
            wrapped.append("".join(prompt_parts))
        if "default" in adapter.get("name"):    
            if jinja_template := self.get_template()["jinja"]:
                jinja_compiled_template = self.jinja_env.from_string(jinja_template)
                messages = []
                if system_instruction:
                    messages.append({
                        'role': 'system',
                        'content': system_instruction
                    })
                messages.extend([
                    {'role': 'user', 'content': user_text},
                    {'role': 'assistant', 'content': ''}
                ])
                wrapped.append(jinja_compiled_template.render(
                    messages=messages,
                    add_generation_prompt=True,
                    bos_token="",
                    eos_token=""
                ))
                if wrapped[1]:
                    return wrapped[1]
        return wrapped[0]