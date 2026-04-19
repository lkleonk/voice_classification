"""
Centralized logging configuration for the voice biomarker pipeline.
Provides consistent formatting and setup across all modules.
"""

import logging
import os
import sys
from typing import Optional

from src.settings import LOGGING


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    add_file_handler: bool = False,
    log_file_path: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the project.
    
    Args:
        name: Logger name (defaults to calling module)
        level: Logging level
        add_file_handler: Whether to add a file handler
        log_file_path: Path for file handler (required if add_file_handler=True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Create formatter with module name for source identification
    formatter = logging.Formatter(
        fmt=LOGGING.FORMAT,
        datefmt=LOGGING.DATEFMT
    )

    # Avoid adding handlers multiple times
    if not logger.handlers:  
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Optional file handler
    if add_file_handler:
        if not log_file_path:
            raise ValueError("log_file_path is required when add_file_handler=True")
        
        # Check if file handler already exists for this path
        log_file_path = os.path.abspath(log_file_path)
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler) and 
                os.path.abspath(handler.baseFilename) == log_file_path
            ):
                break
        else:  # No existing file handler found for this path
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    
    return logger


def setup_run_logger(name: str, log_file_path: str) -> logging.Logger:
    """
    Set up a run-scoped logger that writes to exactly one output log file.

    Reusing the same logger name across multiple training/evaluation runs in one
    Python process would otherwise accumulate file handlers from previous output
    directories and duplicate messages across log files.
    """
    logger = logging.getLogger(name)
    log_file_path = os.path.abspath(log_file_path)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    return setup_logger(name, add_file_handler=True, log_file_path=log_file_path)


def setup_basic_logging(level: int = logging.INFO) -> None:
    """
    Set up basic logging configuration for simple scripts.
    """
    logging.basicConfig(
        level=level,
        format=LOGGING.FORMAT,
        datefmt=LOGGING.DATEFMT,
        stream=sys.stdout
    )

