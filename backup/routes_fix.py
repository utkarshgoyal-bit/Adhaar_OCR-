# routes_ocr_fix.py
"""
Fix for OCR availability check in routes.py
Add this code to the top of your app/api/routes.py file
"""

import logging
logger = logging.getLogger(__name__)

# Proper OCR initialization with detailed logging
def initialize_ocr():
    """Initialize OCR with proper error handling and logging"""
    try:
        from app.ocr.manager import get_ocr_manager
        
        # Get OCR manager
        ocr_manager = get_ocr_manager()
        
        # Check if any engines are available
        available_engines = ocr_manager.get_available_engines()
        
        if available_engines:
            logger.info(f"OCR initialized successfully with engines: {available_engines}")
            return ocr_manager, True
        else:
            logger.warning("OCR manager created but no engines available")
            return ocr_manager, False
            
    except ImportError as e:
        logger.error(f"OCR module import failed: {e}")
        return None, False
    except Exception as e:
        logger.error(f"OCR initialization failed: {e}")
        return None, False

# Initialize OCR
ocr_manager, OCR_AVAILABLE = initialize_ocr()

# Log the final status
if OCR_AVAILABLE:
    logger.info("✅ OCR is ready for document processing")
else:
    logger.warning("❌ OCR is not available - will use fallback mode")

# Function to check OCR status at runtime
def check_ocr_status():
    """Runtime check for OCR availability"""
    global ocr_manager, OCR_AVAILABLE
    
    if ocr_manager is None:
        return False
        
    try:
        # Check if manager still has available engines
        available_engines = ocr_manager.get_available_engines()
        is_available = len(available_engines) > 0
        
        if is_available != OCR_AVAILABLE:
            logger.warning(f"OCR availability changed: {OCR_AVAILABLE} -> {is_available}")
            OCR_AVAILABLE = is_available
            
        return is_available
    except Exception as e:
        logger.error(f"OCR status check failed: {e}")
        return False

# Alternative: Simple direct check (use this if the above doesn't work)
def simple_ocr_check():
    """Simple direct OCR availability check"""
    try:
        from app.ocr.manager import get_ocr_manager
        manager = get_ocr_manager()
        engines = manager.get_available_engines()
        return len(engines) > 0, manager
    except Exception as e:
        logger.error(f"Simple OCR check failed: {e}")
        return False, None

# If you want to replace the initialization entirely, use this:
# OCR_AVAILABLE, ocr_manager = simple_ocr_check()

print(f"Routes OCR Fix Status:")
print(f"  OCR_AVAILABLE: {OCR_AVAILABLE}")
print(f"  ocr_manager: {ocr_manager}")
if ocr_manager:
    print(f"  Available engines: {ocr_manager.get_available_engines()}")
