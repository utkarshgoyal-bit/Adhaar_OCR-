"""
Base Pydantic models for the OCR document parsing service.
Defines the normalized JSON schema structure.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    AADHAAR = "aadhaar"
    PAN = "pan"
    DRIVING_LICENSE = "dl"
    VOTER_ID = "voter_id"


class OCREngine(str, Enum):
    """Supported OCR engines"""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"


class FieldSource(str, Enum):
    """Source of field extraction"""
    OCR = "ocr"
    QR = "qr"
    BARCODE = "barcode"
    MANUAL = "manual"
    PARSER = "parser"       # Added for parser extraction
    FALLBACK = "fallback"   # Added for fallback responses


class FieldValue(BaseModel):
    """Individual field value with metadata"""
    model_config = ConfigDict(extra="forbid")
    
    value: Union[str, Dict[str, Any], None] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="OCR confidence score (0-1)")
    validated: Optional[bool] = Field(None, description="Whether field passed validation")
    masked: Optional[bool] = Field(False, description="Whether value is masked for privacy")
    source: Optional[FieldSource] = Field(None, description="Source of extraction")
    components: Optional[Dict[str, str]] = Field(None, description="Structured components (e.g., address parts)")


class AddressComponents(BaseModel):
    """Structured address components"""
    house_number: Optional[str] = None
    street: Optional[str] = None
    area: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None


class AddressField(FieldValue):
    """Address field with structured components"""
    components: Optional[AddressComponents] = None


class Signals(BaseModel):
    """Quality and validation signals"""
    model_config = ConfigDict(extra="forbid")
    
    checksum_ok: Optional[bool] = Field(None, description="Document checksum validation passed")
    qr_verified: Optional[bool] = Field(None, description="QR code verification passed")
    blur_score: Optional[float] = Field(None, ge=0.0, description="Image blur score (higher = sharper)")
    good_quality: Optional[bool] = Field(None, description="Overall quality assessment")
    language_detected: Optional[List[str]] = Field(default_factory=list, description="Detected languages (ISO codes)")
    rotation_applied: Optional[float] = Field(None, description="Degrees of rotation correction applied")
    document_edges_detected: Optional[bool] = Field(None, description="Full document edges detected")
    suspected_tampering: Optional[bool] = Field(False, description="Potential document tampering detected")


class Meta(BaseModel):
    """Processing metadata"""
    model_config = ConfigDict(extra="forbid")
    
    pages: int = Field(..., ge=1, description="Number of pages processed")
    processing_ms: int = Field(..., ge=0, description="Total processing time in milliseconds")
    engine: str = Field(..., description="OCR engine used")
    upload_hash: str = Field(..., description="SHA256 hash of uploaded file")
    processed_at: datetime = Field(..., description="Processing timestamp (ISO format)")
    field_mappings: Optional[Dict[str, str]] = Field(None, description="Mapping of fields to extraction regions")


class Error(BaseModel):
    """Structured error information"""
    model_config = ConfigDict(extra="forbid")
    
    code: str = Field(..., description="Error code")
    field: Optional[str] = Field(None, description="Field name if error is field-specific")
    message: str = Field(..., description="Human-readable error message")


class Warning(BaseModel):
    """Structured warning information"""
    model_config = ConfigDict(extra="forbid")
    
    code: str = Field(..., description="Warning code")
    field: Optional[str] = Field(None, description="Field name if warning is field-specific")
    message: str = Field(..., description="Human-readable warning message")


class DocumentFields(BaseModel):
    """Base fields common to all documents"""
    model_config = ConfigDict(extra="allow")  # Allow additional fields for specific document types
    
    id_number: Optional[FieldValue] = None
    name: Optional[FieldValue] = None
    dob: Optional[FieldValue] = None
    gender: Optional[FieldValue] = None
    address: Optional[AddressField] = None
    
    # Additional fields for specific document types
    father_name: Optional[FieldValue] = None  # PAN cards
    validity_date: Optional[FieldValue] = None  # DL, Voter ID
    issuing_authority: Optional[FieldValue] = None  # DL


class DocumentResponse(BaseModel):
    """Complete document parsing response"""
    model_config = ConfigDict(extra="forbid")
    
    doc_type: DocumentType = Field(..., description="Type of document processed")
    doc_version: str = Field(default="v1", description="Document schema version")
    parser_version: str = Field(default="1.0.0", description="Parser version used")
    fields: DocumentFields = Field(..., description="Extracted document fields")
    signals: Signals = Field(..., description="Quality and validation signals")
    meta: Meta = Field(..., description="Processing metadata")
    errors: List[Error] = Field(default_factory=list, description="Processing errors")
    warnings: List[Warning] = Field(default_factory=list, description="Processing warnings")


class UploadRequest(BaseModel):
    """Upload request validation"""
    model_config = ConfigDict(extra="forbid")
    
    doc_type: DocumentType = Field(..., description="Type of document being uploaded")


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field("ok", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field("1.0.0", description="Service version")