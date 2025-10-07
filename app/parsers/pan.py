"""
PAN card parser with advanced validation and field extraction.
"""

import re
from typing import Dict, List, Tuple, Any
import logging

from app.schemas.base import DocumentType, FieldValue
from .base import BaseParser, DateParser, NameParser

logger = logging.getLogger(__name__)


class PANParser(BaseParser):
    """Parser for PAN cards with format validation"""
    
    def _get_document_type(self) -> DocumentType:
        return DocumentType.PAN
    
    def _get_field_patterns(self) -> Dict[str, List[str]]:
        """Regex patterns for PAN field extraction"""
        return {
            "pan_number": [
                r'\b([A-Z]{5}\d{4}[A-Z])\b',  # Standard PAN format: ABCDE1234F
                r'PAN\s*(?:No|Number|#)?\s*:?\s*([A-Z]{5}\d{4}[A-Z])',
                r'Permanent\s+Account\s+Number\s*:?\s*([A-Z]{5}\d{4}[A-Z])',
                r'([A-Z]{5}\s?\d{4}\s?[A-Z])',  # With optional spaces
            ],
            "name": [
                r'Name\s*:?\s*([A-Z\s]{3,50})',
                r'([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)',  # All caps name pattern
                r'Signature\s*([A-Z\s]{3,50})',
            ],
            "father_name": [
                r'Father[\'s]*\s+Name\s*:?\s*([A-Z\s]{3,50})',
                r'Father\s*:?\s*([A-Z\s]{3,50})',
                r'S/O\s*:?\s*([A-Z\s]{3,50})',  # Son of
                r'D/O\s*:?\s*([A-Z\s]{3,50})',  # Daughter of
            ],
            "dob": [
                r'DOB\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
                r'Date\s+of\s+Birth\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
                r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',  # Generic date pattern
            ]
        }
    
    def _get_validation_rules(self) -> Dict[str, callable]:
        """Validation functions for PAN fields"""
        return {
            "pan_number": self._validate_pan_number,
            "name": self._validate_pan_name,
            "father_name": self._validate_pan_name,
            "dob": DateParser.parse_date,
        }
    
    def _validate_pan_number(self, pan_str: str) -> Tuple[bool, str, float]:
        """
        Validate PAN number format and structure
        
        PAN format: ABCDE1234F where:
        - First 5 characters: Alphabets
        - Next 4 characters: Numbers  
        - Last character: Alphabet
        - 4th character indicates entity type (P=Individual, etc.)
        
        Returns:
            Tuple of (is_valid, normalized_pan, confidence_score)
        """
        # Clean the PAN number
        clean_pan = re.sub(r'\s+', '', pan_str.strip().upper())
        
        if len(clean_pan) != 10:
            return False, pan_str, 0.0
        
        # Validate PAN format
        if not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', clean_pan):
            return False, pan_str, 0.2
        
        # Additional validations for better confidence
        confidence = 1.0
        
        # Check 4th character (entity type indicator)
        fourth_char = clean_pan[3]
        valid_fourth_chars = ['P', 'F', 'A', 'T', 'B', 'C', 'G', 'J', 'L', 'H', 'K']
        if fourth_char not in valid_fourth_chars:
            confidence *= 0.8  # Reduce confidence but don't reject
        
        # Check if it follows logical patterns
        if fourth_char == 'P':  # Individual
            confidence *= 1.0  # Most common, full confidence
        elif fourth_char in ['F', 'C']:  # Firm/Company
            confidence *= 0.95
        else:
            confidence *= 0.9
        
        return True, clean_pan, confidence
    
    def _validate_pan_name(self, name_str: str) -> Tuple[bool, str, float]:
        """
        Validate and normalize PAN name (usually in ALL CAPS)
        
        Returns:
            Tuple of (is_valid, normalized_name, confidence_score)
        """
        name_str = name_str.strip()
        
        # Remove extra spaces
        name_str = re.sub(r'\s+', ' ', name_str)
        
        # PAN names are typically in ALL CAPS
        if re.match(r'^[A-Z\s\.]+$', name_str) and len(name_str) >= 2:
            # Keep in uppercase for PAN consistency
            confidence = 0.9 if len(name_str.split()) >= 2 else 0.7
            return True, name_str, confidence
        
        # Try to handle mixed case names (convert to uppercase)
        elif re.match(r'^[A-Za-z\s\.]+$', name_str) and len(name_str) >= 2:
            normalized = name_str.upper()
            confidence = 0.8 if len(normalized.split()) >= 2 else 0.6
            return True, normalized, confidence
        
        return False, name_str, 0.3
    
    def _extract_pan_metadata(self, pan_number: str) -> Dict[str, str]:
        """
        Extract metadata from PAN number structure
        
        Args:
            pan_number: Valid 10-character PAN number
            
        Returns:
            Dictionary with extracted details
        """
        if len(pan_number) != 10:
            return {}
        
        details = {}
        
        # 4th character indicates entity type
        entity_type_map = {
            'P': 'Individual',
            'F': 'Firm/LLP', 
            'A': 'Association of Persons (AOP)',
            'T': 'Trust',
            'B': 'Body of Individuals (BOI)',
            'C': 'Company',
            'G': 'Government',
            'J': 'Artificial Juridical Person',
            'L': 'Local Authority',
            'H': 'HUF (Hindu Undivided Family)',
            'K': 'Krish (Special Category)'
        }
        
        fourth_char = pan_number[3]
        if fourth_char in entity_type_map:
            details['entity_type'] = entity_type_map[fourth_char]
        
        # Area code (first 3 characters indicate jurisdiction)
        details['area_code'] = pan_number[:3]
        
        # 5th character (first letter of surname/last name)
        details['surname_initial'] = pan_number[4]
        
        # Sequence number (6th to 9th characters)
        details['sequence_number'] = pan_number[5:9]
        
        # Check digit (last character)
        details['check_digit'] = pan_number[9]
        
        return details
    
    def _validate_pan_with_name(self, pan_number: str, name: str) -> Tuple[bool, float]:
        """
        Cross-validate PAN number with name
        
        Basic check: 5th character of PAN should match first letter of last name
        
        Args:
            pan_number: 10-character PAN number
            name: Full name from PAN card
            
        Returns:
            Tuple of (is_consistent, confidence_modifier)
        """
        if len(pan_number) != 10 or not name:
            return False, 1.0
        
        try:
            # Get 5th character of PAN (should be first letter of surname)
            pan_fifth_char = pan_number[4].upper()
            
            # Get last name (assume last word is surname)
            name_parts = name.strip().upper().split()
            if not name_parts:
                return False, 1.0
            
            surname = name_parts[-1]
            surname_first_char = surname[0]
            
            if pan_fifth_char == surname_first_char:
                return True, 1.1  # Boost confidence
            else:
                # Could be OCR error or different naming convention
                return False, 0.9  # Slight reduction
                
        except (IndexError, AttributeError):
            return False, 1.0
    
    def _create_field_values(self, validated_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Override to handle PAN-specific field mapping and cross-validation"""
        field_values = {}
        
        # Extract PAN number and name for cross-validation
        pan_number = None
        name = None
        
        for field_name, field_data in validated_fields.items():
            if field_name == "pan_number":
                pan_number = field_data["value"]
            elif field_name == "name":
                name = field_data["value"]
        
        # Process each field
        for field_name, field_data in validated_fields.items():
            if field_name == "pan_number":
                # Map to standard id_number
                confidence = field_data["confidence"]
                
                # Apply cross-validation with name if available
                if pan_number and name:
                    is_consistent, confidence_modifier = self._validate_pan_with_name(pan_number, name)
                    confidence *= confidence_modifier
                
                # Extract metadata
                metadata = self._extract_pan_metadata(pan_number) if pan_number else {}
                
                field_values["id_number"] = FieldValue(
                    value=field_data["value"],
                    confidence=min(1.0, confidence),
                    validated=field_data["validated"],
                    source=field_data["source"]
                )
                
                # Store metadata in debug info for later use
                if hasattr(field_data, 'metadata'):
                    field_data['metadata'] = metadata
                
            else:
                # Standard field processing
                field_values[field_name] = FieldValue(
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    validated=field_data["validated"],
                    source=field_data["source"]
                )
        
        return field_values


def create_pan_parser() -> PANParser:
    """Factory function to create PAN parser"""
    return PANParser()


# Test function
def test_pan_parser():
    """Test PAN parser with sample data"""
    parser = create_pan_parser()
    
    # Sample PAN text (simulated OCR output)
    sample_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    
    Permanent Account Number Card
    
    ABCDE1234F
    
    Name: RAHUL KUMAR SHARMA
    Father's Name: SURESH KUMAR SHARMA
    Date of Birth: 21/07/1993
    
    Signature
    """
    
    result = parser.parse(sample_text)
    
    return {
        "confidence": result.confidence_score,
        "fields_extracted": len(result.fields.__dict__) if result.fields else 0,
        "warnings": result.warnings,
        "pan_detected": getattr(result.fields, 'id_number', None) if result.fields else None,
        "name_detected": getattr(result.fields, 'name', None) if result.fields else None
    }


if __name__ == "__main__":
    # Run test if file is executed directly
    test_result = test_pan_parser()
    print("PAN parser test:", test_result)
