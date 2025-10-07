"""
Document parsers module for extracting structured data from Indian government documents.
"""

# Version info
__version__ = "1.0.0"
__author__ = "OCR Document Parser Team"

# Import base classes and utilities
try:
    from .base import BaseParser, ParseResult, DateParser, NameParser
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

# Import registry and expose main functions
try:
    from .registry import ParserRegistry, get_parser_registry, parse_document, get_available_parsers, has_parser_for
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# Import DocumentType from schemas for convenience
try:
    from app.schemas.base import DocumentType
    DOCUMENT_TYPE_AVAILABLE = True
except ImportError:
    DOCUMENT_TYPE_AVAILABLE = False

# Import specific parsers
try:
    from .aadhaar import AadhaarParser, create_aadhaar_parser
    AADHAAR_PARSER_AVAILABLE = True
except ImportError:
    AADHAAR_PARSER_AVAILABLE = False

try:
    from .pan import PANParser, create_pan_parser
    PAN_PARSER_AVAILABLE = True
except ImportError:
    PAN_PARSER_AVAILABLE = False

# Export public interface
__all__ = [
    "__version__",
    "BASE_AVAILABLE",
    "REGISTRY_AVAILABLE", 
    "AADHAAR_PARSER_AVAILABLE",
    "PAN_PARSER_AVAILABLE"
]

# Conditionally add to exports based on availability
if BASE_AVAILABLE:
    __all__.extend([
        "BaseParser", 
        "ParseResult", 
        "DateParser", 
        "NameParser"
    ])

if REGISTRY_AVAILABLE:
    __all__.extend([
        "ParserRegistry",
        "get_parser_registry", 
        "parse_document",
        "get_available_parsers",
        "has_parser_for"
    ])

if DOCUMENT_TYPE_AVAILABLE:
    __all__.extend([
        "DocumentType"
    ])

if AADHAAR_PARSER_AVAILABLE:
    __all__.extend([
        "AadhaarParser",
        "create_aadhaar_parser"
    ])

if PAN_PARSER_AVAILABLE:
    __all__.extend([
        "PANParser",
        "create_pan_parser"
    ])


def get_module_info():
    """Get information about available parsers and components"""
    return {
        "version": __version__,
        "base_classes_available": BASE_AVAILABLE,
        "registry_available": REGISTRY_AVAILABLE,
        "parsers_available": {
            "aadhaar": AADHAAR_PARSER_AVAILABLE,
            "pan": PAN_PARSER_AVAILABLE,
            "driving_license": False,  # TODO: Implement
            "voter_id": False,  # TODO: Implement
        },
        "total_parsers": sum([
            AADHAAR_PARSER_AVAILABLE,
            PAN_PARSER_AVAILABLE
        ])
    }