from jinja2.sandbox import ImmutableSandboxedEnvironment
from pathlib import Path
import json
import re
import time
from typing import Optional, Dict
from .api import KoboldAPI
import requests


class InstructTemplate:
    """ Wraps an instruction and content with the appropriate 
        instruct template.
    """
    def __init__(self, templates_dir: str, url: str):
        self.templates_dir = Path(templates_dir)
        self.api_client = KoboldAPI(url)
        self.url = url
        self.model = self.api_client.get_model()
        self.jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def _normalize(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
        
    def _get_adapter_template(self, model_name: str) -> Optional[Dict]:
        """ Load matching JSON adapter for the model name. """
        if not self.templates_dir.exists():
            return None
        model_name_normalized = self._normalize(model_name)
        best_match = None
        best_match_length = 0
        best_match_version = 0
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
                for name in template["name"]:
                    normalized_name = self._normalize(name)
                    if normalized_name in model_name_normalized:
                        version_match = re.search(r'(\d+)(?:\.(\d+))?', name)
                        current_version = float(f"{version_match.group(1)}.{version_match.group(2) or '0'}") if version_match else 0
                        name_length = len(normalized_name) 
                        if current_version > best_match_version or \
                           (current_version == best_match_version and name_length > best_match_length):
                            best_match = template
                            best_match_length = name_length
                            best_match_version = current_version
                            found_name = name
        except Exception as e:
            print(f"Error reading template files: {str(e)}")
            return None
        #print(f"Chose template: {found_name}")
        return best_match
        
    def _get_props(self) -> Optional[Dict]:
        """ Get template from props endpoint. """
        try:
            if not self.url.endswith('/props'):
                self.url = self.url.rstrip('/') + '/props'
            response = requests.get(self.url)
            response.raise_for_status()
            return response.json().get("chat_template")
        except: 
            return None

    def get_template(self) -> Dict:
        """ Get a template for the appropriate running model. """
        model_name = self.model
        #print(f"Found running model: {model_name}")
        templates = {}
        try:
            templates["adapter"] = self._get_adapter_template(model_name)
            templates["jinja"] = self._get_props()
            return templates
        except:
            print("No template found")
        return None  
        
    def wrap_prompt(self, instruction: str, content: str = "",
                     system_instruction: str = "") -> list:
        """ Format a prompt using the template. 
        
            The instructions and prompt are formatted using the adapter
            and the jinja2 templates and returned on a list.
        """
        templates = self.get_template()
        user_text = f"{content}\n\nINSTRUCTION: {instruction}"
        prompt_parts = []
        wrapped = []
        if adapter := templates["adapter"]:
            if system_instruction is not None:
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
        if jinja_template := templates["jinja"]:
            jinja_compiled_template = self.jinja_env.from_string(jinja_template)
            messages = [
            #{'role': 'system', 'content':},
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': ''}
            ]
            wrapped.append(jinja_compiled_template.render(
                messages=messages,
                add_generation_prompt=True,
                bos_token="",
                eos_token=""
            ))
        return wrapped
        
if __name__ == '__main__':
    instruction = "This is an instruction."
    system_instruction = "The system instruction is a test."
    template_dir = "./templates"
    url = "http://localhost:5001"
    content = "\n\nContent.\n\n"
    template_instance = InstructTemplate(template_dir, url)
    wrapped = template_instance.wrap_prompt(instruction, content, system_instruction)
    print(f"Adapter: \n{wrapped[0]}\n\nJinja:\n{wrapped[1]}")
