# app/ocr/enhanced_preprocessing.py
"""
Enhanced image preprocessing specifically optimized for Indian government ID cards.
Significantly improves OCR accuracy for Aadhaar, PAN, DL, and Voter ID cards.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# Check for OpenCV availability
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - some preprocessing features disabled")


class EnhancedPreprocessor:
    """Enhanced preprocessing for Indian government ID cards"""
    
    def __init__(self):
        self.cv2_available = CV2_AVAILABLE
    
    def preprocess_for_document(self, 
                               image: np.ndarray, 
                               doc_type: str = "generic") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image based on document type
        
        Args:
            image: Input image as numpy array (BGR format)
            doc_type: Document type (aadhaar, pan, dl, voter_id, generic)
        
        Returns:
            Tuple of (preprocessed_image, metadata)
        """
        metadata = {"doc_type": doc_type, "preprocessing_applied": []}
        
        if doc_type == "aadhaar":
            processed, steps = self._preprocess_aadhaar(image)
            metadata["preprocessing_applied"] = steps
            return processed, metadata
        
        elif doc_type == "pan":
            processed, steps = self._preprocess_pan(image)
            metadata["preprocessing_applied"] = steps
            return processed, metadata
        
        else:
            # Generic preprocessing for DL/Voter ID/unknown
            processed, steps = self._preprocess_generic(image)
            metadata["preprocessing_applied"] = steps
            return processed, metadata
    
    def _preprocess_aadhaar(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Aadhaar-specific preprocessing
        Optimized for Hindi+English text with variable contrast
        """
        steps = []
        
        if not self.cv2_available:
            return image, ["opencv_unavailable"]
        
        try:
            # Step 1: Resize if too large (optimal OCR size)
            height, width = image.shape[:2]
            if height > 1200 or width > 1600:
                scale = min(1200/height, 1600/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                steps.append("resize")
            
            # Step 2: Convert to PIL for better enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Step 3: Increase contrast significantly
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.8)
            steps.append("contrast_boost")
            
            # Step 4: Increase sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.2)
            steps.append("sharpness")
            
            # Step 5: Slight brightness adjustment
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.15)
            steps.append("brightness")
            
            # Convert back to OpenCV
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Step 6: Bilateral filter (reduce noise, preserve edges)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            steps.append("bilateral_filter")
            
            # Step 7: Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Step 8: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            steps.append("clahe")
            
            # Step 9: Morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            steps.append("morphology")
            
            # Convert back to BGR for consistency
            final = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            logger.debug(f"Aadhaar preprocessing completed: {steps}")
            return final, steps
            
        except Exception as e:
            logger.error(f"Aadhaar preprocessing failed: {e}")
            return image, ["error"]
    
    def _preprocess_pan(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        PAN card-specific preprocessing
        Optimized for all-caps English text with high contrast
        """
        steps = []
        
        if not self.cv2_available:
            return image, ["opencv_unavailable"]
        
        try:
            # Step 1: Resize if needed
            height, width = image.shape[:2]
            if height > 1000 or width > 1400:
                scale = min(1000/height, 1400/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                steps.append("resize")
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            steps.append("grayscale")
            
            # Step 3: Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            steps.append("gaussian_blur")
            
            # Step 4: Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            steps.append("clahe")
            
            # Step 5: Otsu's thresholding for binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            steps.append("otsu_threshold")
            
            # Step 6: Morphological operations to connect text
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            steps.append("morphology")
            
            # Step 7: Denoise
            binary = cv2.fastNlMeansDenoising(binary, h=10)
            steps.append("denoise")
            
            # Convert to BGR for consistency
            final = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            logger.debug(f"PAN preprocessing completed: {steps}")
            return final, steps
            
        except Exception as e:
            logger.error(f"PAN preprocessing failed: {e}")
            return image, ["error"]
    
    def _preprocess_generic(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Generic preprocessing for DL/Voter ID/unknown documents
        Balanced approach for various document types
        """
        steps = []
        
        if not self.cv2_available:
            return image, ["opencv_unavailable"]
        
        try:
            # Step 1: Resize if too large
            height, width = image.shape[:2]
            if height > 1200 or width > 1600:
                scale = min(1200/height, 1600/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                steps.append("resize")
            
            # Step 2: Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Step 3: Moderate contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            steps.append("contrast")
            
            # Step 4: Moderate sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.8)
            steps.append("sharpness")
            
            # Convert back to OpenCV
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Step 5: Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Step 6: Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            steps.append("clahe")
            
            # Step 7: Slight denoising
            gray = cv2.fastNlMeansDenoising(gray, h=8)
            steps.append("denoise")
            
            # Convert to BGR for consistency
            final = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            logger.debug(f"Generic preprocessing completed: {steps}")
            return final, steps
            
        except Exception as e:
            logger.error(f"Generic preprocessing failed: {e}")
            return image, ["error"]
    
    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Check image quality before processing
        Returns quality metrics
        """
        if not self.cv2_available:
            return {"quality_check": "opencv_unavailable"}
        
        try:
            # Calculate blur score (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Determine quality
            is_good_quality = (
                blur_score > 100 and  # Not too blurry
                50 < brightness < 200 and  # Not too dark/bright
                contrast > 30  # Has reasonable contrast
            )
            
            return {
                "blur_score": float(blur_score),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "good_quality": is_good_quality,
                "resolution": f"{image.shape[1]}x{image.shape[0]}",
                "warnings": self._generate_quality_warnings(blur_score, brightness, contrast)
            }
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {"quality_check": "error", "error": str(e)}
    
    def _generate_quality_warnings(self, 
                                   blur_score: float, 
                                   brightness: float, 
                                   contrast: float) -> list:
        """Generate quality warnings"""
        warnings = []
        
        if blur_score < 100:
            warnings.append("Image is blurry - may affect OCR accuracy")
        
        if brightness < 50:
            warnings.append("Image is too dark")
        elif brightness > 200:
            warnings.append("Image is too bright/overexposed")
        
        if contrast < 30:
            warnings.append("Image has low contrast")
        
        return warnings


# Convenience functions for easy usage
def preprocess_image(image: np.ndarray, 
                    doc_type: str = "generic") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to preprocess image
    
    Args:
        image: Input image as numpy array
        doc_type: Document type (aadhaar, pan, dl, voter_id, generic)
    
    Returns:
        Tuple of (preprocessed_image, metadata)
    """
    preprocessor = EnhancedPreprocessor()
    return preprocessor.preprocess_for_document(image, doc_type)


def check_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Convenience function to check image quality
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Quality metrics dictionary
    """
    preprocessor = EnhancedPreprocessor()
    return preprocessor.check_image_quality(image)


# Test function
def test_enhanced_preprocessing():
    """Test the enhanced preprocessing"""
    
    if not CV2_AVAILABLE:
        return {
            "available": False,
            "message": "OpenCV required for preprocessing"
        }
    
    try:
        # Create test image
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
        
        # Add some text
        cv2.putText(test_image, "AADHAAR TEST", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "1234 5678 9012", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        preprocessor = EnhancedPreprocessor()
        
        # Test quality check
        quality = preprocessor.check_image_quality(test_image)
        
        # Test Aadhaar preprocessing
        aadhaar_processed, aadhaar_steps = preprocessor.preprocess_for_document(
            test_image, "aadhaar"
        )
        
        # Test PAN preprocessing
        pan_processed, pan_steps = preprocessor.preprocess_for_document(
            test_image, "pan"
        )
        
        return {
            "available": True,
            "quality_check": quality,
            "aadhaar_steps": aadhaar_steps,
            "pan_steps": pan_steps,
            "test_successful": True
        }
        
    except Exception as e:
        return {
            "available": True,
            "test_successful": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run test
    result = test_enhanced_preprocessing()
    print("Enhanced preprocessing test:")
    for key, value in result.items():
        print(f"  {key}: {value}")
