"""
OCR module with enhanced preprocessing for Indian government documents.
UPDATED: Clean exports with enhanced preprocessing integration.
"""

import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.1.0-enhanced"

# Check what's available
MANAGER_AVAILABLE = False
ENHANCED_PREPROCESSING_AVAILABLE = False
TESSERACT_AVAILABLE = False

# Import base classes
try:
    from .base import OCREngine, OCRResult, OCRPreprocessor
    BASE_AVAILABLE = True
except ImportError as e:
    BASE_AVAILABLE = False
    logger.warning(f"OCR base classes not available: {e}")

# Import enhanced preprocessing
try:
    from .enhanced_preprocessing import EnhancedPreprocessor, preprocess_image, check_quality
    ENHANCED_PREPROCESSING_AVAILABLE = True
    logger.info("Enhanced preprocessing available")
except ImportError as e:
    ENHANCED_PREPROCESSING_AVAILABLE = False
    logger.warning(f"Enhanced preprocessing not available: {e}")

# Import Tesseract engine
try:
    from .tesseract import TesseractEngine, create_tesseract_engine
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract engine available")
except ImportError as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"Tesseract engine not available: {e}")

# Import OCR manager
try:
    from .manager import OCRManager, get_ocr_manager, extract_text_from_image
    MANAGER_AVAILABLE = True
    logger.info("OCR Manager available")
except ImportError as e:
    MANAGER_AVAILABLE = False
    logger.warning(f"OCR Manager not available: {e}")

# Export public API
__all__ = [
    "__version__",
    "MANAGER_AVAILABLE",
    "ENHANCED_PREPROCESSING_AVAILABLE",
    "TESSERACT_AVAILABLE",
]

# Add base classes if available
if BASE_AVAILABLE:
    __all__.extend([
        "OCREngine",
        "OCRResult",
        "OCRPreprocessor",
    ])

# Add enhanced preprocessing if available
if ENHANCED_PREPROCESSING_AVAILABLE:
    __all__.extend([
        "EnhancedPreprocessor",
        "preprocess_image",
        "check_quality",
    ])

# Add Tesseract if available
if TESSERACT_AVAILABLE:
    __all__.extend([
        "TesseractEngine",
        "create_tesseract_engine",
    ])

# Add manager if available
if MANAGER_AVAILABLE:
    __all__.extend([
        "OCRManager",
        "get_ocr_manager",
        "extract_text_from_image",
    ])


def get_ocr_info():
    """Get information about available OCR components"""
    return {
        "version": __version__,
        "manager_available": MANAGER_AVAILABLE,
        "enhanced_preprocessing": ENHANCED_PREPROCESSING_AVAILABLE,
        "tesseract_available": TESSERACT_AVAILABLE,
        "components": {
            "base_classes": BASE_AVAILABLE,
            "enhanced_preprocessing": ENHANCED_PREPROCESSING_AVAILABLE,
            "tesseract_engine": TESSERACT_AVAILABLE,
            "ocr_manager": MANAGER_AVAILABLE,
        }
    }


# Log initialization status
if MANAGER_AVAILABLE and ENHANCED_PREPROCESSING_AVAILABLE and TESSERACT_AVAILABLE:
    logger.info("✅ OCR module fully initialized with enhanced preprocessing")
else:
    missing = []
    if not MANAGER_AVAILABLE:
        missing.append("OCR Manager")
    if not ENHANCED_PREPROCESSING_AVAILABLE:
        missing.append("Enhanced Preprocessing")
    if not TESSERACT_AVAILABLE:
        missing.append("Tesseract Engine")
    
    if missing:
        logger.warning(f"⚠️ OCR module partially initialized. Missing: {', '.join(missing)}")