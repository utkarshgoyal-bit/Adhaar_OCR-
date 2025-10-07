"""
Base parser interface and common utilities for document parsing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from datetime import datetime
import logging

from app.schemas.base import DocumentFields, FieldValue, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of document parsing with all metadata"""
    fields: Optional[DocumentFields]
    confidence_score: float
    extraction_method: str
    warnings: List[str]
    debug_info: Dict[str, Any]


class BaseParser(ABC):
    """Abstract base class for all document parsers"""
    
    def __init__(self):
        self.document_type = self._get_document_type()
        self.field_patterns = self._get_field_patterns()
        self.validation_rules = self._get_validation_rules()
    
    @abstractmethod
    def _get_document_type(self) -> DocumentType:
        """Return the document type this parser handles"""
        pass
    
    @abstractmethod
    def _get_field_patterns(self) -> Dict[str, List[str]]:
        """Return regex patterns for field extraction"""
        pass
    
    @abstractmethod
    def _get_validation_rules(self) -> Dict[str, callable]:
        """Return validation functions for each field"""
        pass
    
    def supports(self, doc_type: DocumentType) -> bool:
        """Check if this parser supports the given document type"""
        return doc_type == self.document_type
    
    def parse(self, text: str, image_data: Optional[Any] = None) -> ParseResult:
        """
        Main parsing method - extracts structured fields from text
        
        Args:
            text: OCR extracted text or raw text
            image_data: Optional image data for advanced parsing
            
        Returns:
            ParseResult with extracted fields and metadata
        """
        try:
            # Step 1: Clean the text
            clean_text = self._preprocess_text(text)
            
            # Step 2: Extract fields using regex patterns
            extracted_fields = self._extract_fields(clean_text)
            
            # Step 3: Validate extracted fields
            validated_fields = self._validate_fields(extracted_fields)
            
            # Step 4: Convert to FieldValue objects
            final_fields = self._create_field_values(validated_fields)
            
            # Step 5: Calculate overall confidence
            confidence = self._calculate_confidence(final_fields)
            warnings = self._generate_warnings(final_fields)
            
            return ParseResult(
                fields=DocumentFields(**final_fields),
                confidence_score=confidence,
                extraction_method=f"{self.document_type.value}_parser",
                warnings=warnings,
                debug_info={
                    "original_text_length": len(text),
                    "clean_text_length": len(clean_text),
                    "fields_extracted": len(final_fields),
                    "parser_version": "1.0.0"
                }
            )
            
        except Exception as e:
            logger.error(f"Parsing failed for {self.document_type.value}: {e}")
            return ParseResult(
                fields=DocumentFields(),
                confidence_score=0.0,
                extraction_method=f"{self.document_type.value}_parser_error",
                warnings=[f"Parsing failed: {str(e)}"],
                debug_info={"error": str(e)}
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better parsing"""
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere
        clean_text = re.sub(r'[^\w\s\-\/\.\:\,]', ' ', clean_text)
        
        return clean_text
    
    def _extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract fields using regex patterns"""
        extracted = {}
        
        for field_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value = match.group(1) if match.groups() else match.group(0)
                        extracted[field_name] = {
                            "raw_value": value.strip(),
                            "pattern_used": pattern,
                            "confidence": 0.8  # Base confidence for pattern match
                        }
                        break
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {field_name}: {pattern} - {e}")
        
        return extracted
    
    def _validate_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted fields using validation rules"""
        validated = {}
        
        for field_name, field_data in extracted_fields.items():
            if field_name in self.validation_rules:
                try:
                    validation_func = self.validation_rules[field_name]
                    is_valid, normalized_value, confidence_modifier = validation_func(field_data["raw_value"])
                    
                    validated[field_name] = {
                        "value": normalized_value,
                        "confidence": min(1.0, field_data["confidence"] * confidence_modifier),
                        "validated": is_valid,
                        "source": "parser"
                    }
                except Exception as e:
                    logger.warning(f"Validation failed for {field_name}: {e}")
                    validated[field_name] = {
                        "value": field_data["raw_value"],
                        "confidence": field_data["confidence"] * 0.5,
                        "validated": False,
                        "source": "parser"
                    }
            else:
                # No validation rule, accept as-is
                validated[field_name] = {
                    "value": field_data["raw_value"],
                    "confidence": field_data["confidence"],
                    "validated": None,
                    "source": "parser"
                }
        
        return validated
    
    def _create_field_values(self, validated_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Convert validated data to FieldValue objects"""
        field_values = {}
        
        for field_name, field_data in validated_fields.items():
            field_values[field_name] = FieldValue(
                value=field_data["value"],
                confidence=field_data["confidence"],
                validated=field_data["validated"],
                source=field_data["source"]
            )
        
        return field_values
    
    def _calculate_confidence(self, fields: Dict[str, FieldValue]) -> float:
        """Calculate overall parsing confidence"""
        if not fields:
            return 0.0
        
        confidences = [field.confidence for field in fields.values() if field.confidence is not None]
        if not confidences:
            return 0.5
        
        # Weighted average with bonus for validated fields
        total_confidence = 0.0
        total_weight = 0.0
        
        for field in fields.values():
            weight = 1.5 if field.validated else 1.0
            confidence = field.confidence or 0.5
            total_confidence += confidence * weight
            total_weight += weight
        
        return min(1.0, total_confidence / total_weight)
    
    def _generate_warnings(self, fields: Dict[str, FieldValue]) -> List[str]:
        """Generate warnings based on parsing results"""
        warnings = []
        
        if not fields:
            warnings.append("No fields extracted from document")
        
        low_confidence_fields = [
            name for name, field in fields.items() 
            if field.confidence and field.confidence < 0.5
        ]
        
        if low_confidence_fields:
            warnings.append(f"Low confidence fields: {', '.join(low_confidence_fields)}")
        
        unvalidated_fields = [
            name for name, field in fields.items()
            if field.validated is False
        ]
        
        if unvalidated_fields:
            warnings.append(f"Validation failed for: {', '.join(unvalidated_fields)}")
        
        return warnings


class DateParser:
    """Utility class for parsing dates in various Indian formats"""
    
    DATE_PATTERNS = [
        r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
        r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
    ]
    
    @classmethod
    def parse_date(cls, date_str: str) -> Tuple[bool, str, float]:
        """
        Parse date string and return (is_valid, normalized_date, confidence)
        
        Returns:
            Tuple of (is_valid, ISO_date_string, confidence_score)
        """
        date_str = date_str.strip()
        
        for pattern in cls.DATE_PATTERNS:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    if len(match.group(3)) == 4:  # DD/MM/YYYY format
                        day, month, year = match.groups()
                    else:  # YYYY/MM/DD format
                        year, month, day = match.groups()
                    
                    # Create datetime object to validate
                    dt = datetime(int(year), int(month), int(day))
                    return True, dt.strftime('%Y-%m-%d'), 0.9
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Date parsing error: {e}")
                    continue
        
        return False, date_str, 0.0


class NameParser:
    """Utility class for parsing and normalizing names"""
    
    @classmethod
    def normalize_name(cls, name_str: str) -> Tuple[bool, str, float]:
        """
        Normalize name string
        
        Returns:
            Tuple of (is_valid, normalized_name, confidence_score)
        """
        name_str = name_str.strip()
        
        # Remove extra spaces
        name_str = re.sub(r'\s+', ' ', name_str)
        
        # Check if it looks like a name (contains only letters and spaces)
        if re.match(r'^[A-Za-z\s\.]+$', name_str) and len(name_str) >= 2:
            # Title case normalization
            normalized = ' '.join(word.capitalize() for word in name_str.split())
            confidence = 0.9 if len(normalized.split()) >= 2 else 0.7
            return True, normalized, confidence
        
        return False, name_str, 0.3


# Test utility function
def test_base_parser():
    """Quick test function for base parser utilities"""
    
    # Test DateParser
    date_result = DateParser.parse_date("21/07/1993")
    print(f"Date parsing: {date_result}")
    
    # Test NameParser  
    name_result = NameParser.normalize_name("RAHUL KUMAR")
    print(f"Name parsing: {name_result}")
    
    return {
        "date_test": date_result,
        "name_test": name_result
    }


if __name__ == "__main__":
    # Run tests if file is executed directly
    results = test_base_parser()
    print("Base parser tests completed:", results)
