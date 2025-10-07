"""
OCR Manager - Coordinates multiple OCR engines and handles preprocessing.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .base import OCREngine, OCRResult, OCRPreprocessor

logger = logging.getLogger(__name__)

# Import engines with error handling
try:
    from .tesseract import create_tesseract_engine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract engine not available")

# PaddleOCR is optional for now
PADDLEOCR_AVAILABLE = False


class OCRManager:
    """Manages multiple OCR engines with intelligent fallback and preprocessing"""
    
    def __init__(self, 
                 languages: List[str] = None,
                 preferred_engine: str = "tesseract",
                 enable_preprocessing: bool = True):
        
        self.languages = languages or ["eng", "hin"]
        self.preferred_engine = preferred_engine
        self.enable_preprocessing = enable_preprocessing
        self.engines: Dict[str, OCREngine] = {}
        self.preprocessor = OCRPreprocessor()
        
        # Initialize available engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines"""
        engines_loaded = 0
        
        # Try to initialize Tesseract
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
        
        # TODO: Add PaddleOCR when needed
        # if PADDLEOCR_AVAILABLE:
        #     try:
        #         paddleocr_engine = create_paddleocr_engine(self.languages)
        #         if paddleocr_engine.is_available():
        #             self.engines["paddleocr"] = paddleocr_engine
        #             engines_loaded += 1
        #             logger.info("PaddleOCR engine initialized")
        #     except Exception as e:
        #         logger.error(f"Failed to initialize PaddleOCR: {e}")
        
        if engines_loaded == 0:
            logger.error("No OCR engines available! Please install Tesseract.")
        else:
            logger.info(f"OCR Manager initialized with {engines_loaded} engines: {list(self.engines.keys())}")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        return list(self.engines.keys())
    
    def get_best_engine(self, image: np.ndarray = None) -> Optional[str]:
        """
        Select the best OCR engine for the given context
        
        Args:
            image: Input image (for future intelligent selection)
            
        Returns:
            Best engine name or None if no engines available
        """
        available_engines = self.get_available_engines()
        
        if not available_engines:
            return None
        
        # Preference order: preferred_engine -> tesseract -> others
        if self.preferred_engine in available_engines:
            return self.preferred_engine
        
        if "tesseract" in available_engines:
            return "tesseract"
        
        return available_engines[0]  # Fallback to first available
    
    def extract_text(self, 
                    image: np.ndarray, 
                    engine_name: Optional[str] = None,
                    fallback_on_failure: bool = True) -> OCRResult:
        """
        Extract text from image using specified or best available engine
        
        Args:
            image: Input image as numpy array (BGR format)
            engine_name: Specific engine to use (None for auto-selection)
            fallback_on_failure: Try other engines if primary fails
            
        Returns:
            OCRResult with extracted text and metadata
        """
        
        # Select engine
        selected_engine = engine_name or self.get_best_engine(image)
        
        if not selected_engine:
            raise RuntimeError("No OCR engines available")
        
        # Preprocess image if enabled
        processed_image = image
        preprocessing_metadata = {}
        
        if self.enable_preprocessing:
            try:
                processed_image, preprocessing_metadata = self.preprocessor.preprocess_image(image)
                logger.debug(f"Image preprocessing completed: {preprocessing_metadata}")
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {e}")
                processed_image = image
                preprocessing_metadata = {"preprocessing_failed": str(e)}
        
        # Try primary engine
        primary_result = None
        try:
            engine = self.engines[selected_engine]
            primary_result = engine.extract_text(processed_image)
            
            # Enhance result with preprocessing metadata
            if primary_result.bbox_data is None:
                primary_result.bbox_data = []
            
            # Add preprocessing info as metadata
            primary_result.bbox_data.insert(0, {
                'type': 'preprocessing_metadata',
                'data': preprocessing_metadata,
                'engine_used': selected_engine
            })
            
            # If result is good enough, return it
            if primary_result.confidence > 0.2 or not fallback_on_failure:
                logger.info(f"OCR successful with {selected_engine}: confidence={primary_result.confidence:.2f}")
                return primary_result
            
            logger.warning(f"Primary engine {selected_engine} returned low confidence: {primary_result.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Primary OCR engine {selected_engine} failed: {e}")
        
        # Try fallback engines if enabled and primary failed/low confidence
        if fallback_on_failure and len(self.engines) > 1:
            for fallback_engine in self.engines:
                if fallback_engine != selected_engine:
                    try:
                        logger.info(f"Trying fallback OCR engine: {fallback_engine}")
                        engine = self.engines[fallback_engine]
                        result = engine.extract_text(processed_image)
                        
                        if result.confidence > 0.2:
                            logger.info(f"Fallback engine {fallback_engine} succeeded: confidence={result.confidence:.2f}")
                            # Add fallback info to metadata
                            if result.bbox_data is None:
                                result.bbox_data = []
                            result.bbox_data.insert(0, {
                                'type': 'fallback_info',
                                'primary_engine': selected_engine,
                                'fallback_engine': fallback_engine,
                                'preprocessing': preprocessing_metadata
                            })
                            return result
                        
                    except Exception as e:
                        logger.error(f"Fallback engine {fallback_engine} failed: {e}")
        
        # If all engines failed, return the best result we got (or empty result)
        if primary_result:
            logger.warning(f"Returning low-confidence result from {selected_engine}")
            return primary_result
        
        # Complete failure - return empty result
        logger.error("All OCR engines failed")
        return OCRResult(
            text="",
            confidence=0.0,
            language_detected=[],
            processing_time_ms=0,
            engine="all_engines_failed",
            bbox_data=[{
                'type': 'error_info',
                'message': 'All OCR engines failed',
                'available_engines': list(self.engines.keys()),
                'preprocessing': preprocessing_metadata
            }]
        )
    
    def extract_text_with_multiple_engines(self, 
                                         image: np.ndarray,
                                         engines: List[str] = None) -> Dict[str, OCRResult]:
        """
        Extract text using multiple engines for comparison
        
        Args:
            image: Input image
            engines: List of engine names to use (None for all available)
            
        Returns:
            Dictionary mapping engine names to OCRResults
        """
        engines_to_use = engines or self.get_available_engines()
        results = {}
        
        # Preprocess once
        processed_image = image
        preprocessing_metadata = {}
        if self.enable_preprocessing:
            try:
                processed_image, preprocessing_metadata = self.preprocessor.preprocess_image(image)
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {e}")
                preprocessing_metadata = {"error": str(e)}
        
        # Run OCR with each engine
        for engine_name in engines_to_use:
            if engine_name in self.engines:
                try:
                    logger.debug(f"Running OCR with engine: {engine_name}")
                    engine = self.engines[engine_name]
                    result = engine.extract_text(processed_image)
                    
                    # Add engine comparison metadata
                    if result.bbox_data is None:
                        result.bbox_data = []
                    result.bbox_data.insert(0, {
                        'type': 'comparison_metadata',
                        'engine': engine_name,
                        'preprocessing': preprocessing_metadata
                    })
                    
                    results[engine_name] = result
                    
                except Exception as e:
                    logger.error(f"Engine {engine_name} failed: {e}")
                    results[engine_name] = OCRResult(
                        text="",
                        confidence=0.0,
                        language_detected=[],
                        processing_time_ms=0,
                        engine=f"{engine_name}-error",
                        bbox_data=[{'type': 'error', 'message': str(e)}]
                    )
        
        return results
    
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
            "languages": self.languages,
            "health": "healthy" if self.engines else "no_engines",
            "engine_details": self.get_engine_info()
        }


# Global OCR manager instance
_ocr_manager: Optional[OCRManager] = None


def get_ocr_manager(languages: List[str] = None, 
                   preferred_engine: str = "tesseract",
                   enable_preprocessing: bool = True) -> OCRManager:
    """
    Get global OCR manager instance (singleton pattern)
    
    Returns:
        Global OCRManager instance
    """
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager(languages, preferred_engine, enable_preprocessing)
        logger.debug("Created new OCR manager instance")
    return _ocr_manager


def extract_text_from_image(image: np.ndarray, 
                           engine: str = None,
                           languages: List[str] = None) -> OCRResult:
    """
    Convenience function for OCR text extraction
    
    Args:
        image: Input image as numpy array
        engine: Specific engine to use (None for auto-selection)
        languages: Languages to use (None for default eng+hin)
        
    Returns:
        OCRResult with extracted text and metadata
    """
    manager = get_ocr_manager(languages=languages)
    return manager.extract_text(image, engine_name=engine)


# Test function
def test_ocr_manager():
    """Test the OCR manager with all available engines"""
    
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
        test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Try to add text if OpenCV available
        try:
            import cv2
            cv2.putText(test_image, "SAMPLE DOC 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        except ImportError:
            pass
        
        # Test OCR extraction
        result = manager.extract_text(test_image)
        
        return {
            "manager_available": True,
            "engines_available": True,
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
            "test_successful": False,
            "stats": stats,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run comprehensive test if file is executed directly
    result = test_ocr_manager()
    print("OCR Manager test result:")
    for key, value in result.items():
        print(f"  {key}: {value}")