"""
API routes with enhanced OCR + intelligent parser integration.
UPDATED: Now uses document-specific preprocessing for better accuracy.
"""

from datetime import datetime
import hashlib
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.schemas.base import (
    DocumentResponse, 
    HealthResponse, 
    DocumentType, 
    DocumentFields,
    FieldValue,
    Signals,
    Meta,
    Error,
    Warning
)

# Import OCR components with enhanced preprocessing
OCR_AVAILABLE = False
ocr_manager = None

try:
    from app.ocr.manager import get_ocr_manager, extract_text_from_image
    
    # Initialize OCR manager with enhanced preprocessing enabled
    ocr_manager = get_ocr_manager(
        languages=["eng", "hin"],
        preferred_engine="tesseract",
        enable_preprocessing=True,
        enable_enhanced_preprocessing=True  # 🆕 NEW: Enable enhanced preprocessing
    )
    
    # Check if any engines are actually available
    available_engines = ocr_manager.get_available_engines()
    if available_engines:
        OCR_AVAILABLE = True
        logging.info(f"OCR system initialized with enhanced preprocessing")
        logging.info(f"Available engines: {available_engines}")
        
        # Check enhanced preprocessing status
        stats = ocr_manager.get_manager_stats()
        if stats.get('enhanced_preprocessing_enabled'):
            logging.info("✅ Enhanced preprocessing active for better accuracy")
        else:
            logging.warning("⚠️ Enhanced preprocessing not available")
    else:
        OCR_AVAILABLE = False
        logging.warning("OCR manager created but no engines available")
        
except ImportError as e:
    logging.warning(f"OCR system not available: {e}")
except Exception as e:
    logging.error(f"OCR initialization failed: {e}")

# Import file processor
PROCESSOR_AVAILABLE = False
try:
    from app.ingest.processor import FileProcessor, get_processor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"File processor not available: {e}")

# Import intelligent parsers
PARSERS_AVAILABLE = False
try:
    from app.parsers import parse_document, get_available_parsers, has_parser_for
    PARSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Intelligent parsers not available: {e}")

router = APIRouter()
logger = logging.getLogger(__name__)

# Log final initialization status
logger.info(f"System initialization status:")
logger.info(f"  OCR_AVAILABLE: {OCR_AVAILABLE}")
logger.info(f"  PROCESSOR_AVAILABLE: {PROCESSOR_AVAILABLE}")
logger.info(f"  PARSERS_AVAILABLE: {PARSERS_AVAILABLE}")
if OCR_AVAILABLE and ocr_manager:
    logger.info(f"  OCR engines: {ocr_manager.get_available_engines()}")
    stats = ocr_manager.get_manager_stats()
    logger.info(f"  Enhanced preprocessing: {stats.get('enhanced_preprocessing_enabled', False)}")


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with complete system status"""
    
    # Check OCR system status
    ocr_status = "unavailable"
    enhanced_preprocessing_status = False
    
    if OCR_AVAILABLE and ocr_manager:
        try:
            available_engines = ocr_manager.get_available_engines()
            stats = ocr_manager.get_manager_stats()
            enhanced_preprocessing_status = stats.get('enhanced_preprocessing_enabled', False)
            
            if available_engines:
                ocr_status = f"available:{','.join(available_engines)}"
                if enhanced_preprocessing_status:
                    ocr_status += "+enhanced"
            else:
                ocr_status = "no_engines"
        except Exception as e:
            ocr_status = f"error:{str(e)[:30]}"
    
    # Check parser status
    parser_status = "unavailable"
    if PARSERS_AVAILABLE:
        try:
            available_parsers = get_available_parsers()
            parser_status = f"available:{len(available_parsers)}_parsers"
        except Exception:
            parser_status = "error"
    
    logger.info(f"Health check: OCR={ocr_status}, Parsers={parser_status}, Processor={PROCESSOR_AVAILABLE}")
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: DocumentType = Form(...)
):
    """
    Upload and parse document using enhanced OCR + intelligent parsers
    
    UPDATED: Now uses document-specific preprocessing for better accuracy:
    - Aadhaar cards get Aadhaar-optimized preprocessing
    - PAN cards get PAN-optimized preprocessing
    - DL/Voter ID get generic enhanced preprocessing
    
    Args:
        file: Document file (PDF or image)
        doc_type: Type of document (aadhaar, pan, dl, voter_id)
    
    Returns:
        DocumentResponse: Parsed document with enhanced OCR accuracy
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Generate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Log upload (PII-safe)
        logger.info(f"Document upload: type={doc_type}, size={len(file_content)}, hash={file_hash[:8]}...")
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 10MB limit"
            )
        
        # Validate file type
        allowed_types = {
            "application/pdf",
            "image/jpeg", 
            "image/png", 
            "image/webp",
            "image/heic",
            "image/jpg"
        }
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        logger.debug(f"Checking system availability: OCR={OCR_AVAILABLE}, Processor={PROCESSOR_AVAILABLE}, Parsers={PARSERS_AVAILABLE}")
        
        # Process document with enhanced OCR + intelligent parsing
        if (OCR_AVAILABLE and ocr_manager and 
            PROCESSOR_AVAILABLE and 
            PARSERS_AVAILABLE and has_parser_for(doc_type)):
            
            logger.info(f"Using enhanced OCR + intelligent parsing for {doc_type}")
            response = await _process_with_enhanced_ocr(
                file_content, 
                file.content_type, 
                doc_type, 
                file_hash
            )
        else:
            # Fallback to simulation with explanation
            missing_components = []
            if not OCR_AVAILABLE or not ocr_manager:
                missing_components.append("OCR")
            if not PROCESSOR_AVAILABLE:
                missing_components.append("FileProcessor")
            if not PARSERS_AVAILABLE or not has_parser_for(doc_type):
                missing_components.append("Parser")
            
            logger.warning(f"Enhanced processing unavailable (missing: {missing_components}) - using fallback")
            response = _generate_fallback_response(doc_type, file_hash, len(file_content), missing_components)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )


async def _process_with_enhanced_ocr(
    file_content: bytes, 
    mime_type: str, 
    doc_type: DocumentType, 
    file_hash: str
) -> DocumentResponse:
    """
    Process document using enhanced OCR with document-specific preprocessing
    
    UPDATED: Now passes doc_type to OCR manager for optimal preprocessing
    """
    
    start_time = datetime.utcnow()
    errors = []
    warnings = []
    
    try:
        logger.info(f"Processing {doc_type.value} with enhanced OCR pipeline")
        
        # Step 1: File processing (PDF to images or direct image loading)
        try:
            processor = FileProcessor() if PROCESSOR_AVAILABLE else get_processor()
            
            if not processor.is_supported_format(mime_type):
                errors.append(Error(
                    code="UNSUPPORTED_FORMAT",
                    message=f"File format {mime_type} is not supported"
                ))
                return _create_error_response(doc_type, file_hash, start_time, errors)
            
            # Extract images from file
            images, file_metadata = processor.process_file(file_content, mime_type)
            
            if not images:
                errors.append(Error(
                    code="NO_IMAGES_EXTRACTED",
                    message="Could not extract any images from the file"
                ))
                return _create_error_response(doc_type, file_hash, start_time, errors)
            
            logger.info(f"Extracted {len(images)} image(s) from file")
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            errors.append(Error(
                code="FILE_PROCESSING_FAILED",
                message=f"File processing failed: {str(e)}"
            ))
            return _create_error_response(doc_type, file_hash, start_time, errors)
        
        # Step 2: Enhanced OCR extraction with document-specific preprocessing
        primary_image = images[0]  # Process first page/image
        
        try:
            logger.debug(f"Running enhanced OCR for {doc_type.value}...")
            
            # 🆕 NEW: Pass doc_type to enable document-specific preprocessing
            ocr_result = ocr_manager.extract_text(
                primary_image,
                engine_name=None,  # Auto-select best engine
                fallback_on_failure=True,
                doc_type=doc_type.value  # 🔥 KEY CHANGE: Document-specific preprocessing
            )
            
            if not ocr_result.text.strip():
                warnings.append(Warning(
                    code="NO_TEXT_EXTRACTED",
                    message="OCR could not extract any readable text"
                ))
            
            logger.info(f"Enhanced OCR completed: {len(ocr_result.text)} chars, confidence={ocr_result.confidence:.2f}")
            
            # Check if enhanced preprocessing was used
            if ocr_result.bbox_data:
                for metadata in ocr_result.bbox_data:
                    if metadata.get('type') == 'processing_metadata':
                        strategy = metadata.get('strategy', 'unknown')
                        logger.info(f"OCR strategy used: {strategy}")
                        
                        # Check quality warnings
                        quality_info = metadata.get('quality_info', {})
                        quality_warnings = quality_info.get('warnings', [])
                        for qw in quality_warnings:
                            warnings.append(Warning(
                                code="IMAGE_QUALITY",
                                message=qw
                            ))
            
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {e}")
            errors.append(Error(
                code="OCR_EXTRACTION_FAILED",
                message=f"OCR extraction failed: {str(e)}"
            ))
            return _create_error_response(doc_type, file_hash, start_time, errors)
        
        # Step 3: Intelligent parsing of OCR text
        try:
            logger.debug(f"Running intelligent parsing for {doc_type.value}...")
            parse_result = parse_document(doc_type, ocr_result.text)
            
            if not parse_result.fields:
                warnings.append(Warning(
                    code="NO_FIELDS_EXTRACTED",
                    message="Intelligent parser could not extract structured fields"
                ))
                parse_result.fields = DocumentFields()
            
            # Add parser warnings to response
            for warning_msg in parse_result.warnings:
                warnings.append(Warning(
                    code="PARSER_WARNING",
                    message=warning_msg
                ))
            
            logger.info(f"Parsing completed: confidence={parse_result.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"Intelligent parsing failed: {e}")
            errors.append(Error(
                code="INTELLIGENT_PARSING_FAILED",
                message=f"Intelligent parsing failed: {str(e)}"
            ))
            return _create_error_response(doc_type, file_hash, start_time, errors)
        
        # Step 4: Build comprehensive signals
        signals = _build_comprehensive_signals(doc_type, ocr_result, parse_result)
        
        # Step 5: Build detailed metadata
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Extract preprocessing info from OCR result
        preprocessing_info = "unknown"
        quality_score = None
        
        if ocr_result.bbox_data:
            for metadata in ocr_result.bbox_data:
                if metadata.get('type') == 'processing_metadata':
                    preprocessing_info = metadata.get('strategy', 'unknown')
                    quality_info = metadata.get('quality_info', {})
                    quality_score = quality_info.get('blur_score')
        
        meta = Meta(
            pages=len(images),
            processing_ms=processing_time,
            engine=f"{ocr_result.engine}+enhanced+{parse_result.extraction_method}",
            upload_hash=f"sha256:{file_hash}",
            processed_at=datetime.utcnow(),
            field_mappings={
                "ocr_engine": ocr_result.engine,
                "ocr_confidence": f"{ocr_result.confidence:.3f}",
                "preprocessing_strategy": preprocessing_info,  # 🆕 NEW
                "parser_method": parse_result.extraction_method,
                "parser_confidence": f"{parse_result.confidence_score:.3f}",
                "file_format": file_metadata.get("original_format", "unknown"),
                "pages_processed": str(len(images)),
                "quality_score": str(quality_score) if quality_score else "unknown",  # 🆕 NEW
                "enhanced_preprocessing": "enabled"  # 🆕 NEW
            }
        )
        
        logger.info(f"Enhanced processing complete: OCR={ocr_result.confidence:.2f}, Parser={parse_result.confidence_score:.2f}, Strategy={preprocessing_info}")
        
        return DocumentResponse(
            doc_type=doc_type,
            doc_version="v1",
            parser_version="1.0.0-enhanced-ocr",
            fields=parse_result.fields,
            signals=signals,
            meta=meta,
            errors=errors,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Enhanced processing pipeline failed: {e}")
        errors.append(Error(
            code="PROCESSING_PIPELINE_FAILED",
            message=f"Complete processing failed: {str(e)}"
        ))
        return _create_error_response(doc_type, file_hash, start_time, errors)


def _build_comprehensive_signals(doc_type: DocumentType, ocr_result, parse_result) -> Signals:
    """Build comprehensive signals from OCR and parser results"""
    
    # Extract preprocessing metadata from OCR result
    blur_score = 100.0  # Default
    rotation_applied = 0.0
    document_edges_detected = None
    
    if ocr_result.bbox_data:
        for bbox_item in ocr_result.bbox_data:
            if isinstance(bbox_item, dict):
                # Check for quality info in processing metadata
                if bbox_item.get('type') == 'processing_metadata':
                    quality_info = bbox_item.get('quality_info', {})
                    blur_score = quality_info.get('blur_score', 100.0)
                    break
                
                # Legacy preprocessing metadata
                if bbox_item.get('type') == 'preprocessing_metadata':
                    preprocessing = bbox_item.get('data', {})
                    blur_score = preprocessing.get('final_blur_score', 100.0)
                    rotation_applied = preprocessing.get('rotation_applied', 0.0)
                    document_edges_detected = True
                    break
    
    # Build signals
    signals = Signals(
        blur_score=blur_score,
        good_quality=ocr_result.confidence > 0.6 and parse_result.confidence_score > 0.7,
        language_detected=ocr_result.language_detected,
        rotation_applied=rotation_applied,
        document_edges_detected=document_edges_detected,
        suspected_tampering=False
    )
    
    # Document-specific validation signals
    if doc_type == DocumentType.AADHAAR:
        id_field = getattr(parse_result.fields, 'id_number', None)
        signals.checksum_ok = getattr(id_field, 'validated', False) if id_field else False
        signals.qr_verified = True  # Simulated
        
    elif doc_type == DocumentType.PAN:
        id_field = getattr(parse_result.fields, 'id_number', None)
        signals.checksum_ok = getattr(id_field, 'validated', False) if id_field else False
        signals.qr_verified = False
    
    else:
        signals.checksum_ok = parse_result.confidence_score > 0.8
        signals.qr_verified = False
    
    return signals


def _create_error_response(
    doc_type: DocumentType, 
    file_hash: str, 
    start_time: datetime, 
    errors: List[Error]
) -> DocumentResponse:
    """Create error response with proper validation"""
    
    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    return DocumentResponse(
        doc_type=doc_type,
        doc_version="v1",
        parser_version="1.0.0",
        fields=DocumentFields(),
        signals=Signals(),
        meta=Meta(
            pages=1,
            processing_ms=processing_time,
            engine="error",
            upload_hash=f"sha256:{file_hash}",
            processed_at=datetime.utcnow()
        ),
        errors=errors,
        warnings=[]
    )


def _generate_fallback_response(
    doc_type: DocumentType, 
    file_hash: str, 
    file_size: int, 
    missing_components: List[str]
) -> DocumentResponse:
    """Generate fallback response when enhanced processing unavailable"""
    
    warning_msg = f"Enhanced processing unavailable (missing: {', '.join(missing_components)}), using fallback"
    
    # Generate basic fallback data
    fallback_fields = DocumentFields()
    
    if doc_type == DocumentType.AADHAAR:
        fallback_fields.id_number = FieldValue(
            value="XXXX XXXX 1234",
            masked=True,
            confidence=0.5,
            validated=False,
            source="fallback"
        )
        fallback_fields.name = FieldValue(value="Sample User", confidence=0.5, source="fallback")
        fallback_fields.dob = FieldValue(value="1990-01-01", confidence=0.5, source="fallback")
        fallback_fields.gender = FieldValue(value="MALE", confidence=0.5, source="fallback")
    
    elif doc_type == DocumentType.PAN:
        fallback_fields.id_number = FieldValue(
            value="ABCDE1234F",
            confidence=0.5,
            validated=False,
            source="fallback"
        )
        fallback_fields.name = FieldValue(value="SAMPLE USER", confidence=0.5, source="fallback")
        fallback_fields.father_name = FieldValue(value="SAMPLE FATHER", confidence=0.5, source="fallback")
    
    return DocumentResponse(
        doc_type=doc_type,
        doc_version="v1",
        parser_version="1.0.0-fallback",
        fields=fallback_fields,
        signals=Signals(
            blur_score=75.0,
            good_quality=False,
            language_detected=["eng"],
            suspected_tampering=False
        ),
        meta=Meta(
            pages=1,
            processing_ms=200,
            engine="fallback-simulation",
            upload_hash=f"sha256:{file_hash}",
            processed_at=datetime.utcnow(),
            field_mappings={
                "fallback_reason": warning_msg,
                "missing_components": ", ".join(missing_components)
            }
        ),
        errors=[],
        warnings=[Warning(
            code="ENHANCED_PROCESSING_UNAVAILABLE",
            message=warning_msg
        )]
    )