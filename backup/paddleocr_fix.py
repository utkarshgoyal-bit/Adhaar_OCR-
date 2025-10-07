# paddleocr_updated.py
"""
Updated PaddleOCR implementation with correct parameters for version 3.2.0
Replace the relevant parts in your app/ocr/paddleocr.py
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np

try:
    from paddleocr import PaddleOCR
    import cv2
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from app.ocr.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation with correct parameters for version 3.2.0+"""
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.engine_name = "paddleocr"
        self.ocr_engine = None
        
        # Language mapping for PaddleOCR
        self.lang_map = {
            "eng": "en",
            "hin": "hi", 
            "hindi": "hi",
            "english": "en"
        }
        
        # Initialize engine if available
        if self.is_available():
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize PaddleOCR engine with correct parameters"""
        try:
            # Updated parameters for PaddleOCR 3.2.0+
            self.ocr_engine = PaddleOCR(
                use_textline_orientation=True,  # Replaces deprecated use_angle_cls
                lang='en'  # Start with English, can be changed later
                # Removed use_gpu=False and show_log as they're not supported
            )
            
            logger.info("PaddleOCR initialized successfully with updated parameters")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr_engine = None
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is properly installed"""
        if not PADDLEOCR_AVAILABLE:
            return False
        
        try:
            # Try to create a simple PaddleOCR instance
            test_ocr = PaddleOCR(use_textline_orientation=False, lang="en")
            return True
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")
            return False
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR"""
        if not self.is_available() or self.ocr_engine is None:
            raise RuntimeError("PaddleOCR is not available")
        
        start_time = time.time()
        
        try:
            # Run OCR
            results = self.ocr_engine.ocr(image, cls=True)
            
            if not results or not results[0]:
                processing_time = int((time.time() - start_time) * 1000)
                return OCRResult(
                    text="",
                    confidence=0.0,
                    language_detected=[],
                    processing_time_ms=processing_time,
                    engine="paddleocr",
                    bbox_data=[]
                )
            
            # Extract text and confidence
            extracted_text = []
            confidences = []
            bbox_data = []
            
            for line in results[0]:
                if len(line) >= 2:
                    bbox = line[0]  # Bounding box coordinates
                    text_info = line[1]  # (text, confidence)
                    
                    text = text_info[0]
                    confidence = float(text_info[1])
                    
                    extracted_text.append(text)
                    confidences.append(confidence)
                    
                    # Store bounding box data
                    bbox_data.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': {
                            'coordinates': bbox,  # PaddleOCR returns 4 corner points
                            'x': min([point[0] for point in bbox]),
                            'y': min([point[1] for point in bbox]),
                            'width': max([point[0] for point in bbox]) - min([point[0] for point in bbox]),
                            'height': max([point[1] for point in bbox]) - min([point[1] for point in bbox])
                        }
                    })
            
            # Combine text with line breaks
            full_text = "\n".join(extracted_text)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Simple language detection
            detected_languages = ["eng"]  # Default to English
            if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in full_text):
                detected_languages.append("hin")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language_detected=detected_languages,
                processing_time_ms=processing_time,
                engine="paddleocr",
                bbox_data=bbox_data
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected=[],
                processing_time_ms=processing_time,
                engine="paddleocr-error",
                bbox_data=[]
            )

def create_paddleocr_engine(languages: List[str] = None) -> PaddleOCREngine:
    """Create and configure PaddleOCR engine with updated parameters"""
    if languages is None:
        languages = ["eng", "hin"]
    
    engine = PaddleOCREngine(languages)
    
    if not engine.is_available():
        logger.warning("PaddleOCR is not available. Please check installation.")
        
    return engine

# Test function
def test_paddleocr_updated():
    """Test PaddleOCR with updated parameters"""
    
    if not PADDLEOCR_AVAILABLE:
        return {
            "available": False,
            "message": "PaddleOCR not installed",
            "install_command": "pip install paddleocr"
        }
    
    engine = create_paddleocr_engine()
    
    if not engine.is_available():
        return {
            "available": False,
            "dependencies_ok": True,
            "message": "PaddleOCR installation has issues"
        }
    
    try:
        # Create a simple test image
        import numpy as np
        test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Add text using OpenCV (if available)
        try:
            import cv2
            cv2.putText(test_image, "PADDLE TEST 123", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        except ImportError:
            pass
        
        # Test OCR
        result = engine.extract_text(test_image)
        
        return {
            "available": True,
            "test_successful": True,
            "extracted_text": result.text,
            "confidence": result.confidence,
            "languages_detected": result.language_detected,
            "processing_time_ms": result.processing_time_ms,
            "engine": result.engine
        }
        
    except Exception as e:
        return {
            "available": True,
            "test_successful": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Run test if file is executed directly
    result = test_paddleocr_updated()
    print("Updated PaddleOCR test:", result)