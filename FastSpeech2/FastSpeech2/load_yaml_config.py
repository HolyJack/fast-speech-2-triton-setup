import yaml
from importlib import resources


def load_yaml_config(relative_path):
    # Open the config file from the package's configs subdirectory
    with resources.open_text(fFastSpeech2", relative_path) as f:
        config = yaml.safe_load(f)
        return config
