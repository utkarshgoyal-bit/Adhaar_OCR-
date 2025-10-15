"""
OCR Manager - Coordinates OCR engines with enhanced preprocessing for accuracy.
UPDATED: Integrated enhanced preprocessing for Indian ID cards.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .base import OCREngine, OCRResult, OCRPreprocessor

logger = logging.getLogger(__name__)

# Import enhanced preprocessing
try:
    from .enhanced_preprocessing import EnhancedPreprocessor, check_quality
    ENHANCED_PREPROCESSING_AVAILABLE = True
except ImportError:
    ENHANCED_PREPROCESSING_AVAILABLE = False
    logger.warning("Enhanced preprocessing not available")
    
    
    # Import QR scanner
try:
    from .qr_scanner import get_qr_scanner, scan_aadhaar_qr
    QR_SCANNER_AVAILABLE = True
except ImportError:
    QR_SCANNER_AVAILABLE = False
    logger.warning("QR scanner not available - install pyzbar for 100% accuracy")

# Import engines with error handling
try:
    from .tesseract import create_tesseract_engine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract engine not available")

# PaddleOCR removed for lightweight version
PADDLEOCR_AVAILABLE = False


class OCRManager:
    """
    Manages OCR engine with intelligent preprocessing and multi-pass strategy.
    UPDATED: Now uses enhanced preprocessing for better accuracy on Indian IDs.
    """
    
   def __init__(self, 
             languages: List[str] = None,
             preferred_engine: str = "tesseract",
             enable_preprocessing: bool = True,
             enable_enhanced_preprocessing: bool = True,
             enable_qr_scanning: bool = True):  # 🆕 ADD THIS PARAMETER
    
    self.languages = languages or ["eng", "hin"]
    self.preferred_engine = preferred_engine
    self.enable_preprocessing = enable_preprocessing
    self.enable_enhanced_preprocessing = enable_enhanced_preprocessing
    self.enable_qr_scanning = enable_qr_scanning  # 🆕 ADD THIS LINE
    
    self.engines: Dict[str, OCREngine] = {}
    self.preprocessor = OCRPreprocessor()
    self.enhanced_preprocessor = None
    self.qr_scanner = None  # 🆕 ADD THIS LINE
    
    # Initialize enhanced preprocessor if available
    if ENHANCED_PREPROCESSING_AVAILABLE and enable_enhanced_preprocessing:
        self.enhanced_preprocessor = EnhancedPreprocessor()
        logger.info("Enhanced preprocessing enabled for better accuracy")
    
    # 🆕 ADD THIS BLOCK - Initialize QR scanner
    if QR_SCANNER_AVAILABLE and enable_qr_scanning:
        self.qr_scanner = get_qr_scanner()
        if self.qr_scanner.is_available():
            logger.info("QR scanner enabled for Aadhaar cards")
        else:
            logger.warning("QR scanner library available but not functional")
    
    # Initialize available engines
    self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines"""
        engines_loaded = 0
        
        # Initialize Tesseract (primary engine)
        if TESSERACT_AVAILABLE:
            try:
                tesseract_engine = create_tesseract_engine(self.languages)
                if tesseract_engine.is_available():
                    self.engines["tesseract"] = tesseract_engine
                    engines_loaded += 1
                    logger.info("Tesseract OCR engine initialized successfully")
                else:
                    logger.warning("Tesseract OCR engine not available (binary not installed)")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
        else:
            logger.warning("Tesseract module not available")
        
        if engines_loaded == 0:
            logger.error("No OCR engines available! Please install Tesseract.")
        else:
            logger.info(f"OCR Manager initialized with {engines_loaded} engine(s): {list(self.engines.keys())}")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        return list(self.engines.keys())
    
    def get_best_engine(self, image: np.ndarray = None) -> Optional[str]:
        """
        Select the best OCR engine
        
        Args:
            image: Input image (for future intelligent selection)
            
        Returns:
            Best engine name or None if no engines available
        """
        available_engines = self.get_available_engines()
        
        if not available_engines:
            return None
        
        # Preference: tesseract (only engine now)
        if "tesseract" in available_engines:
            return "tesseract"
        
        return available_engines[0]
    
    def extract_text(self, 
                    image: np.ndarray, 
                    engine_name: Optional[str] = None,
                    fallback_on_failure: bool = True,
                    doc_type: Optional[str] = None) -> OCRResult:
        """
        Extract text from image using enhanced preprocessing and multi-pass strategy
        
        Args:
            image: Input image as numpy array (BGR format)
            engine_name: Specific engine to use (None for auto-selection)
            fallback_on_failure: Try multiple strategies if primary fails
            doc_type: Document type for enhanced preprocessing (aadhaar, pan, dl, voter_id)
            
        Returns:
            OCRResult with extracted text and metadata
        """
        
        # Select engine
        selected_engine = engine_name or self.get_best_engine(image)
        
        if not selected_engine:
            raise RuntimeError("No OCR engines available")
        
        # Check image quality first
        quality_info = {}
        if self.enhanced_preprocessor:
            quality_info = self.enhanced_preprocessor.check_image_quality(image)
            logger.debug(f"Image quality: {quality_info}")
            
            # Warn about quality issues
            if not quality_info.get("good_quality", True):
                logger.warning(f"Image quality issues detected: {quality_info.get('warnings', [])}")
        
        # Multi-pass OCR strategy for better accuracy
        best_result = None
        best_confidence = 0.0
        attempts = []
        
        # PASS 1: Quick attempt with basic preprocessing
        try:
            logger.info("Pass 1: Basic preprocessing + OCR")
            basic_result = self._extract_with_basic_preprocessing(
                image, selected_engine
            )
            attempts.append({
                "pass": 1,
                "method": "basic_preprocessing",
                "confidence": basic_result.confidence,
                "text_length": len(basic_result.text)
            })
            
            if basic_result.confidence > best_confidence:
                best_confidence = basic_result.confidence
                best_result = basic_result
            
            # If result is good enough, return early
            if basic_result.confidence > 0.85 and len(basic_result.text) > 20:
                logger.info(f"Pass 1 succeeded with high confidence: {basic_result.confidence:.2f}")
                best_result.bbox_data = best_result.bbox_data or []
                best_result.bbox_data.insert(0, {
                    "type": "processing_metadata",
                    "strategy": "basic_preprocessing",
                    "pass": 1,
                    "quality_info": quality_info,
                    "attempts": attempts
                })
                return best_result
        
        except Exception as e:
            logger.warning(f"Pass 1 failed: {e}")
            attempts.append({"pass": 1, "method": "basic_preprocessing", "error": str(e)})
        
        # PASS 2: Enhanced preprocessing (document-specific)
        if self.enhanced_preprocessor and doc_type:
            try:
                logger.info(f"Pass 2: Enhanced preprocessing for {doc_type}")
                enhanced_result = self._extract_with_enhanced_preprocessing(
                    image, selected_engine, doc_type
                )
                attempts.append({
                    "pass": 2,
                    "method": f"enhanced_preprocessing_{doc_type}",
                    "confidence": enhanced_result.confidence,
                    "text_length": len(enhanced_result.text)
                })
                
                if enhanced_result.confidence > best_confidence:
                    best_confidence = enhanced_result.confidence
                    best_result = enhanced_result
                
                # If result is good enough, return
                if enhanced_result.confidence > 0.75 and len(enhanced_result.text) > 15:
                    logger.info(f"Pass 2 succeeded with confidence: {enhanced_result.confidence:.2f}")
                    best_result.bbox_data = best_result.bbox_data or []
                    best_result.bbox_data.insert(0, {
                        "type": "processing_metadata",
                        "strategy": f"enhanced_preprocessing_{doc_type}",
                        "pass": 2,
                        "quality_info": quality_info,
                        "attempts": attempts
                    })
                    return best_result
            
            except Exception as e:
                logger.warning(f"Pass 2 failed: {e}")
                attempts.append({"pass": 2, "method": "enhanced_preprocessing", "error": str(e)})
        
        # PASS 3: Fallback with generic enhanced preprocessing
        if self.enhanced_preprocessor and fallback_on_failure:
            try:
                logger.info("Pass 3: Generic enhanced preprocessing")
                generic_result = self._extract_with_enhanced_preprocessing(
                    image, selected_engine, "generic"
                )
                attempts.append({
                    "pass": 3,
                    "method": "enhanced_preprocessing_generic",
                    "confidence": generic_result.confidence,
                    "text_length": len(generic_result.text)
                })
                
                if generic_result.confidence > best_confidence:
                    best_confidence = generic_result.confidence
                    best_result = generic_result
                
            except Exception as e:
                logger.warning(f"Pass 3 failed: {e}")
                attempts.append({"pass": 3, "method": "generic_enhanced", "error": str(e)})
        
        # Return best result from all passes
        if best_result:
            logger.info(f"Best result: confidence={best_confidence:.2f}, passes attempted={len(attempts)}")
            best_result.bbox_data = best_result.bbox_data or []
            best_result.bbox_data.insert(0, {
                "type": "processing_metadata",
                "strategy": "multi_pass",
                "best_pass": max(attempts, key=lambda x: x.get("confidence", 0))["pass"],
                "quality_info": quality_info,
                "attempts": attempts
            })
            return best_result
        
        # Complete failure - return empty result
        logger.error("All OCR passes failed")
        return OCRResult(
            text="",
            confidence=0.0,
            language_detected=[],
            processing_time_ms=0,
            engine="all_passes_failed",
            bbox_data=[{
                "type": "error_info",
                "message": "All OCR strategies failed",
                "quality_info": quality_info,
                "attempts": attempts
            }]
        )
    
    def _extract_with_basic_preprocessing(self, 
                                         image: np.ndarray, 
                                         engine_name: str) -> OCRResult:
        """Extract text with basic preprocessing"""
        
        processed_image = image
        preprocessing_metadata = {}
        
        if self.enable_preprocessing:
            try:
                processed_image, preprocessing_metadata = self.preprocessor.preprocess_image(image)
                logger.debug(f"Basic preprocessing completed: {preprocessing_metadata}")
            except Exception as e:
                logger.warning(f"Basic preprocessing failed: {e}")
                preprocessing_metadata = {"error": str(e)}
        
        # Run OCR
        engine = self.engines[engine_name]
        result = engine.extract_text(processed_image)
        
        # Add preprocessing metadata
        if result.bbox_data is None:
            result.bbox_data = []
        result.bbox_data.insert(0, {
            "type": "basic_preprocessing_metadata",
            "data": preprocessing_metadata
        })
        
        return result
    
    def _extract_with_enhanced_preprocessing(self, 
                                            image: np.ndarray, 
                                            engine_name: str,
                                            doc_type: str) -> OCRResult:
        """Extract text with enhanced document-specific preprocessing"""
        
        if not self.enhanced_preprocessor:
            raise RuntimeError("Enhanced preprocessing not available")
        
        # Apply enhanced preprocessing
        processed_image, preprocessing_metadata = self.enhanced_preprocessor.preprocess_for_document(
            image, doc_type
        )
        logger.debug(f"Enhanced preprocessing for {doc_type}: {preprocessing_metadata}")
        
        # Run OCR
        engine = self.engines[engine_name]
        result = engine.extract_text(processed_image)
        
        # Add preprocessing metadata
        if result.bbox_data is None:
            result.bbox_data = []
        result.bbox_data.insert(0, {
            "type": "enhanced_preprocessing_metadata",
            "doc_type": doc_type,
            "data": preprocessing_metadata
        })
        
        return result
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all initialized engines"""
        info = {}
        
        for name, engine in self.engines.items():
            try:
                info[name] = {
                    "available": engine.is_available(),
                    "languages": engine.languages,
                    "engine_name": engine.engine_name,
                    "class": engine.__class__.__name__
                }
            except Exception as e:
                info[name] = {
                    "available": False,
                    "error": str(e),
                    "class": engine.__class__.__name__ if engine else "Unknown"
                }
        
        return info
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get OCR manager statistics and health info"""
        return {
            "total_engines": len(self.engines),
            "available_engines": self.get_available_engines(),
            "preferred_engine": self.preferred_engine,
            "preprocessing_enabled": self.enable_preprocessing,
            "enhanced_preprocessing_enabled": self.enable_enhanced_preprocessing and ENHANCED_PREPROCESSING_AVAILABLE,
            "languages": self.languages,
            "health": "healthy" if self.engines else "no_engines",
            "engine_details": self.get_engine_info()
        }


# Global OCR manager instance
_ocr_manager: Optional[OCRManager] = None


def get_ocr_manager(languages: List[str] = None, 
                   preferred_engine: str = "tesseract",
                   enable_preprocessing: bool = True,
                   enable_enhanced_preprocessing: bool = True) -> OCRManager:
    """
    Get global OCR manager instance (singleton pattern)
    
    Args:
        languages: Languages to use (default: eng+hin)
        preferred_engine: Preferred OCR engine
        enable_preprocessing: Enable basic preprocessing
        enable_enhanced_preprocessing: Enable enhanced document-specific preprocessing
    
    Returns:
        Global OCRManager instance
    """
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager(
            languages, 
            preferred_engine, 
            enable_preprocessing,
            enable_enhanced_preprocessing
        )
        logger.debug("Created new OCR manager instance with enhanced preprocessing")
    return _ocr_manager


def extract_text_from_image(image: np.ndarray, 
                           engine: str = None,
                           languages: List[str] = None,
                           doc_type: str = None) -> OCRResult:
    """
    Convenience function for OCR text extraction with enhanced preprocessing
    
    Args:
        image: Input image as numpy array
        engine: Specific engine to use (None for auto-selection)
        languages: Languages to use (None for default eng+hin)
        doc_type: Document type for enhanced preprocessing (aadhaar, pan, dl, voter_id)
        
    Returns:
        OCRResult with extracted text and metadata
    """
    manager = get_ocr_manager(languages=languages)
    return manager.extract_text(image, engine_name=engine, doc_type=doc_type)


# Test function
def test_ocr_manager():
    """Test the OCR manager with enhanced preprocessing"""
    
    manager = get_ocr_manager()
    stats = manager.get_manager_stats()
    
    if not manager.get_available_engines():
        return {
            "manager_available": True,
            "engines_available": False,
            "stats": stats,
            "message": "No OCR engines available - install Tesseract"
        }
    
    try:
        # Create a simple test image
        import numpy as np
        test_image = np.ones((200, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Try to add text if OpenCV available
        try:
            import cv2
            cv2.putText(test_image, "AADHAAR TEST", (50, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(test_image, "1234 5678 9012", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        except ImportError:
            pass
        
        # Test with enhanced preprocessing
        result = manager.extract_text(test_image, doc_type="aadhaar")
        
        return {
            "manager_available": True,
            "engines_available": True,
            "enhanced_preprocessing": ENHANCED_PREPROCESSING_AVAILABLE,
            "test_successful": True,
            "stats": stats,
            "ocr_result": {
                "text": result.text,
                "confidence": result.confidence,
                "engine": result.engine,
                "processing_time_ms": result.processing_time_ms,
                "languages_detected": result.language_detected
            }
        }
        
    except Exception as e:
        return {
            "manager_available": True,
            "engines_available": True,
            "enhanced_preprocessing": ENHANCED_PREPROCESSING_AVAILABLE,
            "test_successful": False,
            "stats": stats,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run comprehensive test
    result = test_ocr_manager()
    print("OCR Manager test result (with enhanced preprocessing):")
    for key, value in result.items():
        print(f"  {key}: {value}")