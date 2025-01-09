from typing import Dict, Union
from pathlib import Path

from .config import KoboldAPIConfig
from .api import KoboldAPI
from .templates import InstructTemplate

class KoboldAPICore:
    """ Core functionality shared across all LLM tools """
    
    def __init__(self, config_path):
        """ Initialize core services
        
            Args:
                config_path: Path to JSON config file
        """
        self.config = KoboldAPIConfig.from_json(config_path)
        self.api_client = KoboldAPI(
            self.config.api_url, 
            self.config.api_password
        )
        self.template_wrapper = InstructTemplate(
            self.config.templates_directory,
            self.config.api_url
        )
        
    def get_model_info(self) -> Dict[str, Union[str, int]]:
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
            'temperature': self.config.temp,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'rep_pen': self.config.rep_pen,
            'min_p': self.config.min_p
        }