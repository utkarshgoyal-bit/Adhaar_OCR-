"""
Enhanced Tesseract OCR engine with document-specific configurations.
UPDATED: Optimized configs for Indian ID cards (Aadhaar, PAN, DL, Voter ID).
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# Check for Tesseract availability
try:
    import pytesseract
    import cv2
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract dependencies not available")


class TesseractEngine(OCREngine):
    """
    Enhanced Tesseract OCR implementation with document-specific configurations.
    Optimized for Indian government ID cards.
    """
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.engine_name = "tesseract"
        
        # Document-specific Tesseract configurations
        self.document_configs = self._get_document_configs()
        
        # Language mapping for Tesseract
        self.lang_map = {
            "eng": "eng",
            "hin": "hin", 
            "hindi": "hin",
            "english": "eng"
        }
        
        # Verify availability on initialization
        if TESSERACT_AVAILABLE:
            self._verify_installation()
    
    def _get_document_configs(self) -> Dict[str, Dict[str, str]]:
        """
        Get optimized Tesseract configurations for different document types.
        
        PSM Modes (Page Segmentation Mode):
        - PSM 3: Fully automatic (default)
        - PSM 6: Uniform block of text
        - PSM 7: Single line of text
        - PSM 8: Single word
        - PSM 11: Sparse text, find as much as possible
        
        OEM Modes (OCR Engine Mode):
        - OEM 3: Default, based on what is available (LSTM + Legacy)
        """
        
        return {
            # Aadhaar-specific configs
            "aadhaar_default": {
                "config": "--oem 3 --psm 6 -c preserve_interword_spaces=1",
                "lang": "eng+hin",
                "description": "Default Aadhaar config with mixed language support"
            },
            "aadhaar_number": {
                "config": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 ",
                "lang": "eng",
                "description": "Optimized for 12-digit Aadhaar number extraction"
            },
            "aadhaar_name": {
                "config": "--oem 3 --psm 7 -c preserve_interword_spaces=1",
                "lang": "eng+hin",
                "description": "Optimized for name field (single line, mixed language)"
            },
            "aadhaar_address": {
                "config": "--oem 3 --psm 6",
                "lang": "eng+hin",
                "description": "Optimized for multi-line address block"
            },
            
            # PAN card-specific configs
            "pan_default": {
                "config": "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /:-.",
                "lang": "eng",
                "description": "Default PAN config (all caps text)"
            },
            "pan_number": {
                "config": "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "lang": "eng",
                "description": "Optimized for PAN number (5 letters + 4 digits + 1 letter)"
            },
            "pan_name": {
                "config": "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
                "lang": "eng",
                "description": "Optimized for name (all caps, single line)"
            },
            
            # Driving License configs
            "dl_default": {
                "config": "--oem 3 --psm 6 -c preserve_interword_spaces=1",
                "lang": "eng+hin",
                "description": "Default DL config"
            },
            "dl_number": {
                "config": "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/",
                "lang": "eng",
                "description": "Optimized for DL number"
            },
            
            # Voter ID configs
            "voter_default": {
                "config": "--oem 3 --psm 6 -c preserve_interword_spaces=1",
                "lang": "eng+hin",
                "description": "Default Voter ID config"
            },
            "voter_number": {
                "config": "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "lang": "eng",
                "description": "Optimized for EPIC number"
            },
            
            # Generic configs (fallback)
            "generic_default": {
                "config": "--oem 3 --psm 6",
                "lang": "eng+hin",
                "description": "Generic config for unknown documents"
            },
            "generic_sparse": {
                "config": "--oem 3 --psm 11",
                "lang": "eng+hin",
                "description": "Find as much text as possible (sparse)"
            },
            "numbers_only": {
                "config": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
                "lang": "eng",
                "description": "Extract only numbers"
            },
            "letters_only": {
                "config": "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                "lang": "eng",
                "description": "Extract only letters"
            }
        }
    
    def _verify_installation(self):
        """Verify Tesseract installation and available languages"""
        try:
            # Check Tesseract version
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Check available languages
            available_langs = pytesseract.get_languages(config='')
            logger.info(f"Available Tesseract languages: {available_langs}")
            
            # Check if required languages are available
            for lang in self.languages:
                mapped_lang = self.lang_map.get(lang, lang)
                if mapped_lang not in available_langs:
                    logger.warning(f"Language '{mapped_lang}' not available in Tesseract")
            
        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Tesseract is properly installed"""
        if not TESSERACT_AVAILABLE:
            return False
        
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _prepare_languages(self, lang_override: str = None) -> str:
        """Prepare language string for Tesseract"""
        if lang_override:
            return lang_override
        
        tesseract_langs = []
        for lang in self.languages:
            mapped_lang = self.lang_map.get(lang, lang)
            tesseract_langs.append(mapped_lang)
        
        return "+".join(tesseract_langs)
    
    def _get_configs_for_document(self, doc_type: Optional[str]) -> List[tuple]:
        """
        Get prioritized list of configs to try based on document type.
        
        Returns:
            List of (config_name, config_dict) tuples in priority order
        """
        if not doc_type:
            # Generic document - try multiple approaches
            return [
                ("generic_default", self.document_configs["generic_default"]),
                ("generic_sparse", self.document_configs["generic_sparse"]),
            ]
        
        doc_type = doc_type.lower()
        
        if doc_type == "aadhaar":
            return [
                ("aadhaar_default", self.document_configs["aadhaar_default"]),
                ("aadhaar_name", self.document_configs["aadhaar_name"]),
                ("aadhaar_address", self.document_configs["aadhaar_address"]),
                ("generic_default", self.document_configs["generic_default"]),
            ]
        
        elif doc_type == "pan":
            return [
                ("pan_default", self.document_configs["pan_default"]),
                ("pan_name", self.document_configs["pan_name"]),
                ("generic_default", self.document_configs["generic_default"]),
            ]
        
        elif doc_type == "dl" or doc_type == "driving_license":
            return [
                ("dl_default", self.document_configs["dl_default"]),
                ("generic_default", self.document_configs["generic_default"]),
            ]
        
        elif doc_type == "voter_id" or doc_type == "voter":
            return [
                ("voter_default", self.document_configs["voter_default"]),
                ("generic_default", self.document_configs["generic_default"]),
            ]
        
        else:
            # Unknown document type
            return [
                ("generic_default", self.document_configs["generic_default"]),
                ("generic_sparse", self.document_configs["generic_sparse"]),
            ]
    
    def extract_text(self, 
                    image: np.ndarray, 
                    doc_type: Optional[str] = None) -> OCRResult:
        """
        Extract text using Tesseract with document-specific configurations.
        
        Args:
            image: Input image as numpy array (BGR format)
            doc_type: Document type (aadhaar, pan, dl, voter_id) for optimized configs
        
        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.is_available():
            raise RuntimeError("Tesseract OCR is not available")
        
        start_time = time.time()
        
        try:
            # Get document-specific configs to try
            configs_to_try = self._get_configs_for_document(doc_type)
            
            best_result = None
            best_confidence = 0.0
            all_attempts = []
            
            # Try each config in priority order
            for config_name, config_dict in configs_to_try:
                try:
                    config_str = config_dict["config"]
                    lang_str = config_dict["lang"]
                    
                    logger.debug(f"Trying config '{config_name}': {config_dict['description']}")
                    
                    # Extract text with current config
                    text = pytesseract.image_to_string(
                        image, 
                        lang=lang_str, 
                        config=config_str
                    ).strip()
                    
                    # Get detailed data for confidence calculation
                    data = pytesseract.image_to_data(
                        image, 
                        lang=lang_str, 
                        config=config_str,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    avg_confidence = avg_confidence / 100.0
                    
                    # Track attempt
                    all_attempts.append({
                        "config_name": config_name,
                        "text_length": len(text),
                        "confidence": avg_confidence,
                        "description": config_dict["description"]
                    })
                    
                    logger.debug(f"Config '{config_name}': {len(text)} chars, confidence {avg_confidence:.3f}")
                    
                    # Keep best result (prioritize confidence, then text length)
                    score = avg_confidence + (min(len(text), 100) / 1000)  # Slight bonus for longer text
                    best_score = best_confidence + (min(len(best_result["text"]) if best_result else 0, 100) / 1000)
                    
                    if score > best_score:
                        best_confidence = avg_confidence
                        best_result = {
                            'text': text,
                            'data': data,
                            'config_name': config_name,
                            'confidence': avg_confidence,
                            'lang': lang_str
                        }
                    
                    # Early exit if we got a great result
                    if avg_confidence > 0.85 and len(text) > 20:
                        logger.info(f"Config '{config_name}' achieved high confidence, stopping early")
                        break
                
                except Exception as e:
                    logger.debug(f"Config '{config_name}' failed: {e}")
                    all_attempts.append({
                        "config_name": config_name,
                        "error": str(e)
                    })
                    continue
            
            # Use best result
            if best_result:
                text = best_result['text']
                data = best_result['data']
                avg_confidence = best_result['confidence']
                config_used = best_result['config_name']
                lang_used = best_result['lang']
                
                logger.info(f"Best config: '{config_used}' with confidence {avg_confidence:.3f}")
            else:
                # Complete failure
                logger.error("All Tesseract configs failed")
                processing_time = int((time.time() - start_time) * 1000)
                return OCRResult(
                    text="",
                    confidence=0.0,
                    language_detected=[],
                    processing_time_ms=processing_time,
                    engine="tesseract-all-failed",
                    bbox_data=[{
                        "type": "error",
                        "message": "All configs failed",
                        "attempts": all_attempts
                    }]
                )
            
            # Detect languages in the text
            detected_languages = self._detect_languages_in_text(text)
            
            # Extract bounding box data for advanced parsing
            bbox_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Filter low confidence
                    bbox_data.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i], 
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'level': data['level'][i],
                        'word_num': data['word_num'][i]
                    })
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Add config metadata
            bbox_data.insert(0, {
                "type": "tesseract_config_metadata",
                "config_used": config_used,
                "doc_type": doc_type,
                "language": lang_used,
                "all_attempts": all_attempts,
                "configs_tried": len(all_attempts)
            })
            
            logger.info(f"Tesseract extraction complete: {len(text)} chars, {avg_confidence:.2f} confidence, {processing_time}ms")
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                language_detected=detected_languages,
                processing_time_ms=processing_time,
                engine=f"tesseract-{lang_used}-{config_used}",
                bbox_data=bbox_data
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected=[],
                processing_time_ms=processing_time,
                engine="tesseract-error",
                bbox_data=[{"type": "error", "message": str(e)}]
            )
    
    def _detect_languages_in_text(self, text: str) -> List[str]:
        """Detect languages present in extracted text"""
        detected_langs = []
        
        # Check for Devanagari (Hindi) characters
        if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text):
            detected_langs.append("hin")
        
        # Check for ASCII (English) characters
        if any(char.isascii() and char.isalpha() for char in text):
            detected_langs.append("eng")
        
        return detected_langs if detected_langs else ["eng"]
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """Simple text extraction without detailed metadata"""
        if not self.is_available():
            return ""
        
        try:
            lang_string = self._prepare_languages()
            config = self.document_configs["generic_default"]["config"]
            return pytesseract.image_to_string(
                image, 
                lang=lang_string, 
                config=config
            ).strip()
        except Exception as e:
            logger.error(f"Simple Tesseract extraction failed: {e}")
            return ""


def create_tesseract_engine(languages: List[str] = None) -> TesseractEngine:
    """Factory function to create and configure enhanced Tesseract engine"""
    if languages is None:
        languages = ["eng", "hin"]
    
    engine = TesseractEngine(languages)
    
    if not engine.is_available():
        logger.warning("Tesseract OCR is not available. Please install tesseract and pytesseract.")
        logger.info("Installation instructions:")
        logger.info("  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin")
        logger.info("  macOS: brew install tesseract tesseract-lang")
        logger.info("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        
    return engine


# Test function
def test_enhanced_tesseract():
    """Test the enhanced Tesseract engine"""
    
    if not TESSERACT_AVAILABLE:
        return {
            "available": False,
            "message": "Tesseract dependencies not installed"
        }
    
    engine = create_tesseract_engine()
    
    if not engine.is_available():
        return {
            "available": False,
            "dependencies_ok": True,
            "tesseract_binary": False,
            "message": "Tesseract binary not installed"
        }
    
    try:
        # Create test images for different document types
        import numpy as np
        
        # Test Aadhaar-like image
        aadhaar_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(aadhaar_img, "Name: Rahul Kumar", (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(aadhaar_img, "1234 5678 9012", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Test PAN-like image
        pan_img = np.ones((150, 400, 3), dtype=np.uint8) * 255
        cv2.putText(pan_img, "ABCDE1234F", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(pan_img, "JOHN DOE", (50, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Test Aadhaar
        aadhaar_result = engine.extract_text(aadhaar_img, doc_type="aadhaar")
        
        # Test PAN
        pan_result = engine.extract_text(pan_img, doc_type="pan")
        
        return {
            "available": True,
            "test_successful": True,
            "aadhaar_test": {
                "text": aadhaar_result.text,
                "confidence": aadhaar_result.confidence,
                "engine": aadhaar_result.engine
            },
            "pan_test": {
                "text": pan_result.text,
                "confidence": pan_result.confidence,
                "engine": pan_result.engine
            },
            "document_configs": len(engine.document_configs)
        }
        
    except Exception as e:
        return {
            "available": True,
            "test_successful": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run test
    result = test_enhanced_tesseract()
    print("Enhanced Tesseract test:")
    for key, value in result.items():
        print(f"  {key}: {value}")