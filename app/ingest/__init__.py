"""
File ingestion and processing module.
"""

try:
    from .processor import FileProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False

__all__ = [
    "PROCESSOR_AVAILABLE"
]

if PROCESSOR_AVAILABLE:
    __all__.append("FileProcessor")