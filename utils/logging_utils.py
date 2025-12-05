import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to the log level."""
    
    COLOR_MAP = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record):
        # Save original levelname
        orig_levelname = record.levelname
        # Colorize levelname
        color = self.COLOR_MAP.get(record.levelno, Colors.WHITE)
        record.levelname = f"{color}{orig_levelname}{Colors.RESET}"
        
        result = super().format(record)
        
        # Restore original levelname to not affect other handlers
        record.levelname = orig_levelname
        return result

class PlainFormatter(logging.Formatter):
    """Plain formatter for file output."""
    pass

def tee_stdout_stderr(log_dir: str | Path = 'logs', script_basename: Optional[str] = None, time_format: str = '[%Y-%m-%d %H:%M:%S]') -> str:
    """
    Configures the root logger to write to a file and the console.
    Returns the path to the log file.
    """
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    
    if script_basename is None:
        script_basename = Path(sys.argv[0]).stem or 'script'
        
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = base / f"{ts}_{script_basename}.log"
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler (Plain text)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    # Format: [LEVEL] time : message
    file_handler.setFormatter(PlainFormatter("[%(levelname)s] %(asctime)s %(message)s", datefmt=time_format))
    logger.addHandler(file_handler)
    
    # Console Handler (Colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter("[%(levelname)s] %(asctime)s %(message)s", datefmt=time_format))
    logger.addHandler(console_handler)
    
    # Log initial info (use concise time format)
    logging.info(f"===== {script_basename} start {datetime.now().strftime(time_format)} =====")
    logging.info(f"cmd: {' '.join(sys.argv)}")
    
    return str(log_path)

def log_info(message: str) -> None:
    """Logs an info message."""
    if message and str(message).strip():
        logging.info(message)

def log_warn(message: str) -> None:
    """Logs a warning message."""
    if message and str(message).strip():
        logging.warning(message)

def log_error(message: str) -> None:
    """Logs an error message."""
    if message and str(message).strip():
        logging.error(message)
