"""
Simplified file processing module for handling PDF and image inputs.
This version works without heavy dependencies for basic testing.
"""

import logging
import io
from typing import List, Union, Tuple, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles file format conversion and image extraction"""
    
    SUPPORTED_IMAGE_FORMATS = {
        'image/jpeg', 'image/jpg', 'image/png', 
        'image/webp', 'image/heic', 'image/bmp', 'image/tiff'
    }
    
    SUPPORTED_PDF_FORMAT = {'application/pdf'}
    
    @classmethod
    def is_supported_format(cls, mime_type: str) -> bool:
        """Check if file format is supported"""
        return mime_type.lower() in (cls.SUPPORTED_IMAGE_FORMATS | cls.SUPPORTED_PDF_FORMAT)
    
    @classmethod
    def is_pdf(cls, mime_type: str) -> bool:
        """Check if file is PDF"""
        return mime_type.lower() in cls.SUPPORTED_PDF_FORMAT
    
    @classmethod
    def is_image(cls, mime_type: str) -> bool:
        """Check if file is image"""
        return mime_type.lower() in cls.SUPPORTED_IMAGE_FORMATS
    
    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
        """
        Convert PDF to list of images
        
        Args:
            pdf_bytes: PDF file content as bytes
            dpi: Resolution for conversion (default 200 DPI)
            
        Returns:
            List of images as numpy arrays (BGR format)
        """
        if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE or not CV2_AVAILABLE:
            raise RuntimeError("PDF processing not available. Install PyMuPDF, Pillow, and OpenCV.")
        
        images = []
        
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document.load_page(page_num)
                
                # Create transformation matrix for DPI
                zoom = dpi / 72  # 72 is default DPI
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert PIL to OpenCV format (BGR)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                images.append(cv_image)
            
            pdf_document.close()
            logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise RuntimeError(f"Failed to convert PDF: {e}")
        
        return images
    
    @staticmethod
    def bytes_to_image(image_bytes: bytes, mime_type: str) -> np.ndarray:
        """
        Convert image bytes to OpenCV format
        
        Args:
            image_bytes: Image file content as bytes
            mime_type: MIME type of the image
            
        Returns:
            Image as numpy array (BGR format)
        """
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            raise RuntimeError("Image processing not available. Install Pillow and OpenCV.")
        
        try:
            # Handle different image formats
            if mime_type.lower() == 'image/heic':
                # HEIC requires special handling
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    logger.warning("HEIC support not available. Install pillow-heif.")
                    raise RuntimeError("HEIC format requires pillow-heif package")
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Converted {mime_type} image to OpenCV format: {cv_image.shape}")
            return cv_image
            
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            raise RuntimeError(f"Failed to convert image: {e}")
    
    @classmethod
    def process_file(cls, file_bytes: bytes, mime_type: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Process uploaded file and extract images
        
        Args:
            file_bytes: File content as bytes
            mime_type: MIME type of the file
            
        Returns:
            Tuple of (list_of_images, metadata)
        """
        if not cls.is_supported_format(mime_type):
            raise ValueError(f"Unsupported file format: {mime_type}")
        
        metadata = {
            "original_format": mime_type,
            "file_size_bytes": len(file_bytes),
            "pages_extracted": 0
        }
        
        images = []
        
        try:
            if cls.is_pdf(mime_type):
                # Process PDF
                images = cls.pdf_to_images(file_bytes)
                metadata["pages_extracted"] = len(images)
                metadata["processing_type"] = "pdf_conversion"
                
            elif cls.is_image(mime_type):
                # Process image
                image = cls.bytes_to_image(file_bytes, mime_type)
                images = [image]
                metadata["pages_extracted"] = 1
                metadata["processing_type"] = "direct_image"
                
            else:
                raise ValueError(f"Unsupported format: {mime_type}")
            
            # Add image metadata
            if images:
                first_image = images[0]
                metadata["image_dimensions"] = {
                    "height": first_image.shape[0],
                    "width": first_image.shape[1],
                    "channels": first_image.shape[2] if len(first_image.shape) > 2 else 1
                }
            
            logger.info(f"File processing completed: {metadata}")
            return images, metadata
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise


# Simple fallback processor for when dependencies are missing
class FallbackProcessor:
    """Basic processor that creates dummy images when dependencies missing"""
    
    @staticmethod
    def create_dummy_image() -> np.ndarray:
        """Create a dummy image for testing when libraries unavailable"""
        # Create a simple 100x100 gray image
        return np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    @classmethod
    def process_file_fallback(cls, file_bytes: bytes, mime_type: str) -> Tuple[List[np.ndarray], Dict]:
        """Fallback processing when main libraries unavailable"""
        logger.warning("Using fallback processor - dependencies not available")
        
        metadata = {
            "original_format": mime_type,
            "file_size_bytes": len(file_bytes),
            "pages_extracted": 1,
            "processing_type": "fallback_dummy"
        }
        
        # Return dummy image
        dummy_image = cls.create_dummy_image()
        return [dummy_image], metadata


# Auto-select processor based on available dependencies
def get_processor():
    """Get appropriate processor based on available dependencies"""
    if CV2_AVAILABLE and PIL_AVAILABLE:
        return FileProcessor()
    else:
        logger.warning("Main dependencies not available, using fallback processor")
        return FallbackProcessor()


# Make FallbackProcessor methods available at module level for import
def process_file_fallback(file_bytes: bytes, mime_type: str):
    """Module level fallback function"""
    return FallbackProcessor.process_file_fallback(file_bytes, mime_type)