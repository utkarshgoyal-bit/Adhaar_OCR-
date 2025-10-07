"""
Base OCR interface and preprocessing utilities.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for OpenCV availability
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available - image preprocessing disabled")


@dataclass
class OCRResult:
    """OCR extraction result with confidence and language info"""
    text: str
    confidence: float
    language_detected: List[str]
    processing_time_ms: int
    engine: str
    bbox_data: Optional[List[Dict[str, Any]]] = None  # Bounding box data for advanced parsing


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["eng", "hin"]
        self.engine_name = self.__class__.__name__.lower()
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            OCRResult with extracted text and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if OCR engine is properly installed and available"""
        pass
    
    def supports_language(self, lang_code: str) -> bool:
        """Check if engine supports given language code"""
        return lang_code in self.languages


class OCRPreprocessor:
    """Image preprocessing utilities for better OCR accuracy"""
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """
        Calculate Laplacian variance to measure image blur
        Higher values = sharper image
        """
        if not OPENCV_AVAILABLE:
            return 100.0  # Default good score
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Blur score calculation failed: {e}")
            return 100.0
    
    @staticmethod
    def auto_rotate(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Auto-rotate image to correct orientation
        Returns rotated image and rotation angle applied
        """
        if not OPENCV_AVAILABLE:
            return image, 0.0
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate most common angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    # Normalize to -45 to 45 degrees
                    if angle > 45:
                        angle = angle - 90
                    elif angle < -45:
                        angle = angle + 90
                    angles.append(angle)
                
                if angles:
                    # Use median angle for rotation
                    rotation_angle = np.median(angles)
                    
                    if abs(rotation_angle) > 1:  # Only rotate if significant
                        height, width = image.shape[:2]
                        center = (width // 2, height // 2)
                        
                        # Create rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        
                        # Apply rotation
                        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                               flags=cv2.INTER_CUBIC, 
                                               borderMode=cv2.BORDER_REPLICATE)
                        
                        return rotated, rotation_angle
            
        except Exception as e:
            logger.warning(f"Auto-rotation failed: {e}")
        
        return image, 0.0
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if not OPENCV_AVAILABLE:
            return image
        
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    @staticmethod
    def resize_for_ocr(image: np.ndarray, target_height: int = 1000) -> np.ndarray:
        """Resize image to optimal size for OCR (around 300 DPI equivalent)"""
        if not OPENCV_AVAILABLE:
            return image
        
        try:
            height, width = image.shape[:2]
            
            if height > target_height:
                # Calculate new width maintaining aspect ratio
                ratio = target_height / height
                new_width = int(width * ratio)
                
                # Use high-quality interpolation for downscaling
                return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            
        except Exception as e:
            logger.warning(f"Image resize failed: {e}")
        
        return image
    
    @classmethod
    def preprocess_image(cls, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing pipeline
        
        Returns:
            Tuple of (processed_image, processing_metadata)
        """
        metadata = {}
        
        # Calculate original blur score
        original_blur = cls.calculate_blur_score(image)
        metadata["original_blur_score"] = original_blur
        
        # Resize if too large
        processed = cls.resize_for_ocr(image)
        
        # Auto-rotate
        processed, rotation_angle = cls.auto_rotate(processed)
        metadata["rotation_applied"] = rotation_angle
        
        # Enhance contrast
        processed = cls.enhance_contrast(processed)
        
        # Calculate final blur score
        final_blur = cls.calculate_blur_score(processed)
        metadata["final_blur_score"] = final_blur
        metadata["good_quality"] = final_blur > 100  # Threshold for good quality
        
        return processed, metadata


# Test function for OCR base functionality
def test_ocr_base():
    """Test OCR base classes and preprocessing"""
    
    if not OPENCV_AVAILABLE:
        return {
            "opencv_available": False,
            "message": "OpenCV not available - install opencv-python to test preprocessing"
        }
    
    # Create a simple test image
    import numpy as np
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128  # Gray image
    
    # Test preprocessing
    preprocessor = OCRPreprocessor()
    processed, metadata = preprocessor.preprocess_image(test_image)
    
    return {
        "opencv_available": True,
        "test_image_shape": test_image.shape,
        "processed_image_shape": processed.shape,
        "preprocessing_metadata": metadata,
        "blur_score": preprocessor.calculate_blur_score(test_image)
    }


if __name__ == "__main__":
    # Run test if file is executed directly
    result = test_ocr_base()
    print("OCR base test result:", result)