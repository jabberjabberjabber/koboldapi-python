from typing import Dict, Union
from pathlib import Path

from .api import KoboldAPI
from .templates import InstructTemplate
from .config import KoboldAPIConfig

class KoboldAPICore:
    """ Core functionality shared across all LLM tools """
    
    def __init__(self, config_dict: dict):
        """Initialize with a config dict"""
        
        
        self.api_client = KoboldAPI(
            config_dict.get('api_url', 'http://localhost:5000'),
            config_dict.get('api_password', '')
        )
        self.template_wrapper = InstructTemplate(
            config_dict.get('templates_directory', './templates'),
            config_dict.get('api_url', 'http://localhost:5000')
        )
        # Store the whole config dict if you need other values later
        self.config_dict = config_dict
        
            
    def get_model_info(self):
        """ Get current model details """
        return {
            'name': self.api_client.get_model(),
            'context_length': self.api_client.get_max_context_length(),
            'version': self.api_client.get_version()
        }

    def validate_connection(self) -> bool:
        """ Test API connection """
        try:
            self.api_client.get_version()
            return True
        except Exception:
            return False
            
    def get_generation_params(self) -> Dict[str, Union[float, int]]:
        """ Get current generation parameters """
        return {
            'temperature': KoboldAPIConfig.temp,
            'top_k': KoboldAPIConfig.top_k,
            'top_p': KoboldAPIConfig.top_p,
            'rep_pen': KoboldAPIConfig.rep_pen,
            'min_p': KoboldAPIConfig.min_p
        }