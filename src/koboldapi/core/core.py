from typing import Dict, Union
from pathlib import Path
from typing import Optional, Dict, List, Union
from .api import KoboldAPI
from .templates import InstructTemplate

class KoboldAPICore:
    """ Core functionality shared across all LLM tools """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 generation_params: Optional[Dict] = None,
                 templates_directory: Optional[str] = None):
        """Initialize with direct parameters"""
        self.api_client = KoboldAPI(
            api_url=api_url,
            api_password=api_password,
            generation_params=generation_params,
            templates_directory=templates_directory
        )
        
        self.template_wrapper = InstructTemplate(
            api_url=api_url,
            templates_dir=templates_directory
        )
    def get_model_info(self):
        """ Get current model details """
        return {
            'name': self.api_client.get_model(),
            'context_length': self.api_client.get_max_context_length(),
            'version': self.api_client.get_version(),
        }
    
    def get_generation_params(self) -> Dict[str, Union[float, int]]:
        """ Get current generation parameters """
        return self.api_client.generation_params
     
