"""
Logging Configuration for PyPSA-GB

Simplified, reliable logging setup for Snakemake workflows.

USAGE IN SCRIPTS:
-----------------
    from logging_config import setup_logging, get_log_path
    
    # Best practice - pass snakemake.log[0] directly:
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "my_script"
    logger = setup_logging(log_path)
    
    # Or use the helper function:
    logger = setup_logging(get_log_path(fallback="my_script"))

KEY INSIGHT: 
snakemake.log[0] contains the exact path where the log should go.
Scripts MUST pass this path to setup_logging() - we cannot reliably detect it
from within the logging module itself.
"""

import logging
import sys
import os
import time
import functools
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Union

# =============================================================================
# SUPPRESS NOISY LIBRARY WARNINGS
# =============================================================================

# Suppress pandas FutureWarnings about downcasting (comes from PyPSA io.py)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*infer_objects.*')


def get_log_path(fallback: str = None) -> Optional[str]:
    """
    Get the Snakemake log path from the global snakemake object.
    
    This is the RECOMMENDED way to get the log path in scripts.
    Call this at the top of your script's main block.
    
    Parameters
    ----------
    fallback : str, optional
        Fallback log name if not running under Snakemake
        
    Returns
    -------
    str or None
        Path to log file, or fallback name, or None
        
    Example
    -------
    >>> from logging_config import setup_logging, get_log_path
    >>> logger = setup_logging(get_log_path(fallback="my_script"))
    """
    try:
        # Try to access snakemake object from calling frame
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_globals = frame.f_back.f_globals
            if 'snakemake' in caller_globals:
                snakemake_obj = caller_globals['snakemake']
                if hasattr(snakemake_obj, 'log') and snakemake_obj.log:
                    return str(snakemake_obj.log[0])
    except Exception:
        pass
    
    return fallback


def setup_logging(
    log_path: Union[str, Path, None] = None,
    log_level: str = "INFO",
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging with file output for Snakemake scripts.
    
    SIMPLIFIED DESIGN:
    - If log_path is provided (from snakemake.log[0]), log to that file
    - If log_path looks like a path (has / or .log), use as file path
    - Otherwise, create logs/{log_path}.log
    - Console output is enabled by default
    
    Parameters
    ----------
    log_path : str, Path, or None
        Path to log file. Best practice: pass snakemake.log[0]
        Can also be just a name like "solve_network" -> logs/solve_network.log
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_console : bool
        Whether to also log to console (default True)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Examples
    --------
    >>> # Best practice - use Snakemake log path directly
    >>> log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "my_script"
    >>> logger = setup_logging(log_path)
    
    >>> # Simpler with helper
    >>> from logging_config import setup_logging, get_log_path
    >>> logger = setup_logging(get_log_path(fallback="my_script"))
    """
    # Determine log file path
    if log_path is None:
        log_path = "pypsa_gb"
    
    log_path = str(log_path)
    
    # Determine if this looks like a path or just a name
    if os.path.isabs(log_path):
        # Absolute path - use as-is
        log_file = Path(log_path)
    elif '/' in log_path or '\\' in log_path or log_path.endswith('.log'):
        # Relative path or has .log extension - use as-is
        log_file = Path(log_path)
    else:
        # Just a name - put in logs directory
        log_file = Path("logs") / f"{log_path}.log"
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger with name based on the log file stem
    logger_name = log_file.stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers (prevents duplicate messages on re-import)
    logger.handlers.clear()
    
    # Configure format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)
    
    # File handler - ALWAYS add when we have a log path
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"WARNING: Could not create log file {log_file}: {e}", file=sys.stderr)
    
    # Console handler - optional but recommended for visibility
    if log_to_console:
        # Only add if there's no StreamHandler already on this logger
        has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if not has_stream:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False
    
    # Log where we're logging to (helps debug logging issues!)
    logger.debug(f"Logging initialized: file={log_file}, level={log_level}")
    
    return logger


def get_snakemake_logger(fallback_name: str = "snakemake_script") -> logging.Logger:
    """
    Convenience function to get a logger for Snakemake scripts.
    
    Automatically tries to find snakemake.log[0] from the calling frame.
    
    Parameters
    ----------
    fallback_name : str
        Name to use if snakemake object not found
        
    Returns
    -------
    logging.Logger
        Configured logger
        
    Example
    -------
    >>> from logging_config import get_snakemake_logger
    >>> logger = get_snakemake_logger("solve_network")
    """
    log_path = get_log_path(fallback=fallback_name)
    return setup_logging(log_path)

def log_dataframe_info(df, logger, name: str):
    """Helper function to log DataFrame information consistently."""
    logger.info(f"{name} shape: {df.shape}")
    logger.debug(f"{name} columns: {list(df.columns)}")
    logger.debug(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if df.empty:
        logger.warning(f"{name} is empty!")

def log_network_info(network, logger):
    """Helper function to log PyPSA network information consistently."""
    logger.info(f"Network: {network.name}")
    logger.info(f"  - Buses: {len(network.buses)}")
    logger.info(f"  - Lines: {len(network.lines)}")
    logger.info(f"  - Transformers: {len(network.transformers)}")
    logger.info(f"  - Links: {len(network.links)}")
    logger.info(f"  - Loads: {len(network.loads)}")
    logger.info(f"  - Snapshots: {len(network.snapshots)}")

def log_stage_timing(stage_name: str, logger=None):
    """
    Decorator to log execution time for a processing stage.
    
    Parameters
    ----------
    stage_name : str
        Name of the stage being timed (e.g., "Load renewable sites")
    logger : logging.Logger, optional
        Logger instance to use. If None, uses module logger.
        
    Returns
    -------
    Callable
        Decorated function with timing logging
        
    Examples
    --------
    >>> @log_stage_timing("Load DUKES data", logger)
    >>> def load_dukes_generators(year):
    ...     # function code
    ...     return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from decorator param or try to find in function scope
            log = logger
            if log is None:
                # Try to get logger from calling module
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_globals = frame.f_back.f_globals
                    if 'logger' in caller_globals:
                        log = caller_globals['logger']
                    else:
                        log = logging.getLogger(__name__)
            
            start = time.time()
            log.info(f"Starting {stage_name}...")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                log.info(f"✅ {stage_name} complete in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                log.error(f"❌ {stage_name} failed after {elapsed:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

def log_stage_summary(stage_times: Dict[str, float], logger, title: str = "STAGE EXECUTION SUMMARY"):
    """
    Log summary table of stage execution times.
    
    Parameters
    ----------
    stage_times : dict
        Dictionary mapping stage names to execution times (seconds)
    logger : logging.Logger
        Logger instance to use
    title : str
        Title for the summary section
        
    Examples
    --------
    >>> stage_times = {
    ...     "Load sites": 2.34,
    ...     "Map to buses": 1.87,
    ...     "Add generators": 2.16
    ... }
    >>> log_stage_summary(stage_times, logger)
    """
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
    
    if not stage_times:
        logger.warning("No stage times recorded")
        return
    
    total_time = sum(stage_times.values())
    
    for stage, elapsed in stage_times.items():
        if total_time > 0:
            pct = (elapsed / total_time) * 100
            logger.info(f"{stage:45s} {elapsed:8.2f}s ({pct:5.1f}%)")
        else:
            logger.info(f"{stage:45s} {elapsed:8.2f}s")
    
    logger.info("-" * 80)
    logger.info(f"{'TOTAL':45s} {total_time:8.2f}s (100.0%)")
    logger.info("=" * 80)

def log_execution_summary(logger, script_name: str, start_time, inputs=None, outputs=None, context=None):
    """Log a standardized execution summary for Snakemake scripts."""
    duration = time.time() - start_time
    
    logger.info(f"=== {script_name} Execution Summary ===")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    if inputs:
        logger.info(f"Input files: {len(inputs) if hasattr(inputs, '__len__') else 'N/A'}")
        if isinstance(inputs, dict):
            for key, inp in inputs.items():
                if isinstance(inp, (str, Path)):
                    inp_path = Path(inp)
                    if inp_path.exists():
                        size_mb = inp_path.stat().st_size / 1024**2
                        logger.info(f"  - {inp_path.name}: {size_mb:.2f} MB")
        else:
            for inp in (inputs if isinstance(inputs, (list, tuple)) else [inputs]):
                if isinstance(inp, (str, Path)):
                    inp_path = Path(inp)
                    if inp_path.exists():
                        size_mb = inp_path.stat().st_size / 1024**2
                        logger.info(f"  - {inp_path.name}: {size_mb:.2f} MB")
    
    if outputs:
        logger.info(f"Output files: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
        if isinstance(outputs, dict):
            for key, out in outputs.items():
                if isinstance(out, (str, Path)):
                    out_path = Path(out)
                    if out_path.exists():
                        size_mb = out_path.stat().st_size / 1024**2
                        logger.info(f"  - {out_path.name}: {size_mb:.2f} MB")
        else:
            for out in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
                if isinstance(out, (str, Path)):
                    out_path = Path(out)
                    if out_path.exists():
                        size_mb = out_path.stat().st_size / 1024**2
                        logger.info(f"  - {out_path.name}: {size_mb:.2f} MB")
    
    if context:
        logger.info("Context information:")
        for key, value in context.items():
            logger.info(f"  - {key}: {value}")
    
    logger.info(f"=== {script_name} Completed Successfully ===")

# Configure matplotlib and other libraries to reduce noise
def configure_library_logging():
    """Configure logging for common libraries to reduce noise."""
    # Reduce matplotlib noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Reduce pandas warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    
    # Reduce PyPSA logging noise (e.g., messages about networks not being optimized)
    # Set pypsa.networks to ERROR so informational/warning messages are suppressed
    # project-wide. Individual scripts can still raise logging levels if needed.
    logging.getLogger('pypsa.networks').setLevel(logging.ERROR)
    # Also reduce the root pypsa logger to WARNING to avoid excessive detail
    logging.getLogger('pypsa').setLevel(logging.WARNING)

# Initialize library logging configuration
try:
    import pandas as pd
    configure_library_logging()
except ImportError:
    pass

