"""
OCR Module with proper dependency checking and error handling
Place this in your app/ocr/ directory
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class OCRResult:
    """OCR result with text and metadata"""
    def __init__(self, text: str, confidence: float = 0.0, engine: str = "unknown"):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.languages_detected = []
        self.processing_time_ms = 0

class OCREngine:
    """Base OCR engine interface"""
    def __init__(self):
        self.available = False
        self.error_msg = None
    
    def is_available(self) -> bool:
        return self.available
    
    def get_error(self) -> Optional[str]:
        return self.error_msg
    
    def process(self, image: Image.Image, languages: List[str] = None) -> OCRResult:
        raise NotImplementedError

class TesseractEngine(OCREngine):
    """Tesseract OCR Engine"""
    def __init__(self):
        super().__init__()
        self._check_availability()
    
    def _check_availability(self):
        """Check if Tesseract is available"""
        try:
            import pytesseract
            import subprocess
            
            # Test if tesseract binary is accessible
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.available = True
                logger.info("Tesseract OCR engine is available")
            else:
                self.error_msg = f"Tesseract binary failed: {result.stderr}"
                logger.warning(f"Tesseract binary check failed: {self.error_msg}")
        
        except ImportError:
            self.error_msg = "pytesseract package not installed"
            logger.warning("pytesseract package not found")
        except FileNotFoundError:
            self.error_msg = "tesseract binary not found in PATH"
            logger.warning("tesseract binary not found")
        except subprocess.TimeoutExpired:
            self.error_msg = "tesseract binary timeout"
            logger.warning("tesseract binary timeout")
        except Exception as e:
            self.error_msg = f"tesseract check failed: {str(e)}"
            logger.warning(f"Tesseract availability check failed: {e}")
    
    def process(self, image: Image.Image, languages: List[str] = None) -> OCRResult:
        """Process image with Tesseract"""
        if not self.available:
            raise RuntimeError(f"Tesseract not available: {self.error_msg}")
        
        import pytesseract
        
        start_time = time.time()
        
        try:
            # Prepare language string
            lang_str = "+".join(languages) if languages else "eng"
            
            # Get text and confidence data
            text = pytesseract.image_to_string(image, lang=lang_str)
            
            # Try to get detailed data for confidence
            try:
                data = pytesseract.image_to_data(image, lang=lang_str, output_type=pytesseract.Output.DICT)
                # Calculate average confidence (excluding -1 values)
                confidences = [c for c in data['conf'] if c > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                avg_confidence = avg_confidence / 100.0  # Convert to 0-1 scale
            except:
                avg_confidence = 0.7  # Default confidence if detailed data fails
            
            processing_time = (time.time() - start_time) * 1000
            
            result = OCRResult(text.strip(), avg_confidence, f"tesseract-{lang_str}")
            result.languages_detected = languages if languages else ["eng"]
            result.processing_time_ms = processing_time
            
            logger.debug(f"Tesseract OCR completed in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise RuntimeError(f"Tesseract processing failed: {str(e)}")

class PaddleOCREngine(OCREngine):
    """PaddleOCR Engine"""
    def __init__(self):
        super().__init__()
        self.ocr_instance = None
        self._check_availability()
    
    def _check_availability(self):
        """Check if PaddleOCR is available"""
        try:
            from paddleocr import PaddleOCR
            
            # Try to initialize PaddleOCR (this might download models on first run)
            self.ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.available = True
            logger.info("PaddleOCR engine is available")
            
        except ImportError:
            self.error_msg = "paddleocr package not installed"
            logger.warning("PaddleOCR package not found")
        except Exception as e:
            self.error_msg = f"PaddleOCR initialization failed: {str(e)}"
            logger.warning(f"PaddleOCR initialization failed: {e}")
    
    def process(self, image: Image.Image, languages: List[str] = None) -> OCRResult:
        """Process image with PaddleOCR"""
        if not self.available:
            raise RuntimeError(f"PaddleOCR not available: {self.error_msg}")
        
        start_time = time.time()
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            results = self.ocr_instance.ocr(img_array, cls=True)
            
            # Extract text and calculate average confidence
            all_text = []
            all_confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            all_text.append(text)
                            all_confidences.append(confidence)
            
            # Combine results
            combined_text = "\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            result = OCRResult(combined_text, avg_confidence, "paddleocr-en")
            result.languages_detected = ["eng"]  # PaddleOCR doesn't provide language detection
            result.processing_time_ms = processing_time
            
            logger.debug(f"PaddleOCR completed in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            raise RuntimeError(f"PaddleOCR processing failed: {str(e)}")

class OCRManager:
    """OCR Manager that handles multiple engines"""
    
    def __init__(self):
        self.engines = {}
        self.preferred_order = ['tesseract', 'paddleocr']
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines"""
        logger.info("Initializing OCR engines...")
        
        # Initialize Tesseract
        tesseract = TesseractEngine()
        self.engines['tesseract'] = tesseract
        
        # Initialize PaddleOCR
        paddleocr = PaddleOCREngine()
        self.engines['paddleocr'] = paddleocr
        
        # Log availability
        available_engines = [name for name, engine in self.engines.items() if engine.is_available()]
        unavailable_engines = [name for name, engine in self.engines.items() if not engine.is_available()]
        
        if available_engines:
            logger.info(f"Available OCR engines: {available_engines}")
        else:
            logger.warning("No OCR engines are available!")
        
        if unavailable_engines:
            logger.warning(f"Unavailable OCR engines: {unavailable_engines}")
            for name in unavailable_engines:
                logger.warning(f"  {name}: {self.engines[name].get_error()}")
    
    def is_any_available(self) -> bool:
        """Check if any OCR engine is available"""
        return any(engine.is_available() for engine in self.engines.values())
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        return [name for name, engine in self.engines.items() if engine.is_available()]
    
    def get_unavailable_engines(self) -> Dict[str, str]:
        """Get dictionary of unavailable engines and their error messages"""
        return {name: engine.get_error() 
                for name, engine in self.engines.items() 
                if not engine.is_available()}
    
    def process(self, image: Image.Image, languages: List[str] = None, engine: str = None) -> OCRResult:
        """
        Process image with OCR
        
        Args:
            image: PIL Image to process
            languages: List of languages (e.g., ['eng', 'hin'])
            engine: Specific engine to use, or None for auto-selection
        
        Returns:
            OCRResult with extracted text and metadata
        """
        
        if not self.is_any_available():
            unavailable = self.get_unavailable_engines()
            error_details = "; ".join(f"{k}: {v}" for k, v in unavailable.items())
            raise RuntimeError(f"No OCR engines available. Errors: {error_details}")
        
        # Determine which engine to use
        if engine:
            if engine not in self.engines:
                raise ValueError(f"Unknown OCR engine: {engine}")
            if not self.engines[engine].is_available():
                raise RuntimeError(f"Requested OCR engine '{engine}' is not available: {self.engines[engine].get_error()}")
            selected_engine = engine
        else:
            # Auto-select first available engine from preferred order
            selected_engine = None
            for preferred in self.preferred_order:
                if preferred in self.engines and self.engines[preferred].is_available():
                    selected_engine = preferred
                    break
            
            if not selected_engine:
                available = self.get_available_engines()
                selected_engine = available[0]  # Take any available engine
        
        logger.debug(f"Using OCR engine: {selected_engine}")
        
        # Process with selected engine
        try:
            result = self.engines[selected_engine].process(image, languages)
            logger.info(f"OCR completed with {selected_engine} in {result.processing_time_ms:.1f}ms")
            return result
        except Exception as e:
            logger.error(f"OCR processing failed with {selected_engine}: {e}")
            
            # Try fallback to other engines if auto-selection failed
            if not engine:  # Only try fallback if engine wasn't explicitly requested
                available = self.get_available_engines()
                for fallback_engine in available:
                    if fallback_engine != selected_engine:
                        try:
                            logger.warning(f"Trying fallback OCR engine: {fallback_engine}")
                            result = self.engines[fallback_engine].process(image, languages)
                            logger.info(f"Fallback OCR completed with {fallback_engine}")
                            return result
                        except Exception as fallback_e:
                            logger.error(f"Fallback OCR {fallback_engine} also failed: {fallback_e}")
            
            # If we get here, all attempts failed
            raise RuntimeError(f"All OCR engines failed. Last error: {str(e)}")

# Global OCR manager instance
_ocr_manager = None

def get_ocr_manager() -> OCRManager:
    """Get or create global OCR manager instance"""
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager()
    return _ocr_manager

def process_image(image: Image.Image, languages: List[str] = None, engine: str = None) -> OCRResult:
    """
    Convenience function to process image with OCR
    
    Args:
        image: PIL Image to process
        languages: List of language codes (e.g., ['eng', 'hin'])
        engine: Specific OCR engine to use (optional)
    
    Returns:
        OCRResult with extracted text and metadata
    """
    manager = get_ocr_manager()
    return manager.process(image, languages, engine)

def check_ocr_availability() -> Dict[str, Any]:
    """
    Check OCR availability and return status
    
    Returns:
        Dictionary with availability status and details
    """
    manager = get_ocr_manager()
    
    return {
        "any_available": manager.is_any_available(),
        "available_engines": manager.get_available_engines(),
        "unavailable_engines": manager.get_unavailable_engines(),
        "total_engines": len(manager.engines)
    }

# Example usage:
if __name__ == "__main__":
    # Check availability
    status = check_ocr_availability()
    print("OCR Status:", status)
    
    if status["any_available"]:
        print("✅ OCR is working!")
        
        # Create a test image
        test_img = Image.new('RGB', (300, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.text((10, 30), "Test OCR Text", fill='black')
        
        try:
            result = process_image(test_img, languages=['eng'])
            print(f"OCR Result: '{result.text}' (confidence: {result.confidence:.2f})")
        except Exception as e:
            print(f"OCR Test failed: {e}")
    else:
        print("❌ No OCR engines available!")
        for engine, error in status["unavailable_engines"].items():
            print(f"  {engine}: {error}")
