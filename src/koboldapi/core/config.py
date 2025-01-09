from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class LLMConfig:
    """ Configuration for LLM tools and API connections """
    api_url: str
    api_password: str
    templates_directory: str
    translation_language: str = "English"
    text_completion: bool = False
    temp: float = 0.2
    top_k: int = 0
    top_p: float = 1.0
    rep_pen: float = 1.1
    min_p: float = 0.02

    @classmethod
    def from_json(cls, path: str):
        """ Load configuration from JSON file.
            Expects a JSON object with the same field names as the class.
        """
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, path: str):
        """ Save configuration to JSON file """
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)