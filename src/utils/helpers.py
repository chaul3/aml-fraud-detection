import yaml
import logging
import os
from typing import Dict, Any

class ConfigLoader:
    """Configuration loader for AML fraud detection system."""
    
    @staticmethod
    def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return {}

class Logger:
    """Logger setup for AML fraud detection system."""
    
    @staticmethod
    def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
        """Set up logger with specified configuration."""
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

def create_directory_structure(base_path: str) -> None:
    """Create the standard directory structure for the project."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "models",
        "logs",
        "reports",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    print(f"Created directory structure in {base_path}")

def validate_data_schema(df, required_columns: list) -> bool:
    """Validate that DataFrame has required columns for AML processing."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True
