from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

class ConfigManager:
    """Handles configuration loading and validation."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        suffix = self.config_path.suffix.lower()
        if suffix == '.json':
            with open(self.config_path) as f:
                self.config = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

def setup_logging(log_path: Optional[Union[str, Path]] = None,
                 level: int = logging.INFO) -> None:
    """Setup logging configuration.
    
    Args:
        log_path: Path to log file. If None, logs to console only.
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def validate_path(path: Union[str, Path], create: bool = False) -> Path:
    """Validate and optionally create a directory path.
    
    Args:
        path: Directory path to validate
        create: If True, create directory if it doesn't exist
        
    Returns:
        Validated Path object
    """
    path = Path(path)
    if create and not path.exists():
        path.mkdir(parents=True)
    elif not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path

def get_timestamp() -> str:
    """Get current timestamp in standard format.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_number(value: float, precision: int = 2) -> str:
    """Format number with specified precision.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"