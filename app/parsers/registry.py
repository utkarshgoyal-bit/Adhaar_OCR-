"""
Parser registry for managing document parsers.
"""

from typing import Dict, Optional, List, Any
import logging

from app.schemas.base import DocumentType
from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for document parsers with plugin-style architecture"""
    
    def __init__(self):
        self.parsers: Dict[DocumentType, BaseParser] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize all available parsers"""
        parsers_loaded = 0
        
        # Try to register Aadhaar parser
        try:
            from .aadhaar import create_aadhaar_parser
            aadhaar_parser = create_aadhaar_parser()
            self.parsers[DocumentType.AADHAAR] = aadhaar_parser
            parsers_loaded += 1
            logger.info("Aadhaar parser registered successfully")
        except ImportError as e:
            logger.warning(f"Aadhaar parser not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Aadhaar parser: {e}")
        
        # Try to register PAN parser
        try:
            from .pan import create_pan_parser
            pan_parser = create_pan_parser()
            self.parsers[DocumentType.PAN] = pan_parser
            parsers_loaded += 1
            logger.info("PAN parser registered successfully")
        except ImportError as e:
            logger.warning(f"PAN parser not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize PAN parser: {e}")
        
        # TODO: Add DL and Voter ID parsers when implemented
        # try:
        #     from .driving_license import create_dl_parser
        #     dl_parser = create_dl_parser()
        #     self.parsers[DocumentType.DRIVING_LICENSE] = dl_parser
        #     parsers_loaded += 1
        # except ImportError:
        #     pass
        
        logger.info(f"Parser registry initialized with {parsers_loaded} parsers")
    
    def get_parser(self, doc_type: DocumentType) -> Optional[BaseParser]:
        """Get parser for specific document type"""
        return self.parsers.get(doc_type)
    
    def get_available_parsers(self) -> List[DocumentType]:
        """Get list of available parser document types"""
        return list(self.parsers.keys())
    
    def has_parser(self, doc_type: DocumentType) -> bool:
        """Check if parser is available for document type"""
        return doc_type in self.parsers
    
    def parse_document(self, doc_type: DocumentType, text: str, image_data: Any = None) -> ParseResult:
        """
        Parse document using appropriate parser
        
        Args:
            doc_type: Type of document to parse
            text: OCR extracted text or raw text
            image_data: Optional image data for advanced parsing
            
        Returns:
            ParseResult with extracted fields and metadata
        """
        parser = self.get_parser(doc_type)
        
        if not parser:
            logger.warning(f"No parser available for document type: {doc_type.value}")
            return ParseResult(
                fields=None,
                confidence_score=0.0,
                extraction_method="no_parser",
                warnings=[f"No parser available for {doc_type.value}"],
                debug_info={
                    "error": "Parser not found",
                    "available_parsers": [p.value for p in self.get_available_parsers()]
                }
            )
        
        try:
            logger.debug(f"Parsing {doc_type.value} document with {parser.__class__.__name__}")
            result = parser.parse(text, image_data)
            
            # Add registry metadata to debug info
            if result.debug_info:
                result.debug_info.update({
                    "parser_class": parser.__class__.__name__,
                    "registry_version": "1.0.0"
                })
            
            logger.info(f"Parsed {doc_type.value} with confidence: {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Parsing failed for {doc_type.value}: {e}")
            return ParseResult(
                fields=None,
                confidence_score=0.0,
                extraction_method=f"{doc_type.value}_parser_error",
                warnings=[f"Parsing failed: {str(e)}"],
                debug_info={
                    "error": str(e),
                    "parser_class": parser.__class__.__name__ if parser else "None"
                }
            )
    
    def register_parser(self, parser: BaseParser):
        """
        Register a new parser (for future extensibility)
        
        Args:
            parser: Instance of BaseParser subclass
        """
        doc_type = parser._get_document_type()
        self.parsers[doc_type] = parser
        logger.info(f"Registered parser for {doc_type.value}: {parser.__class__.__name__}")
    
    def unregister_parser(self, doc_type: DocumentType):
        """Remove a parser from registry"""
        if doc_type in self.parsers:
            del self.parsers[doc_type]
            logger.info(f"Unregistered parser for {doc_type.value}")
    
    def get_parser_info(self) -> Dict[str, Dict]:
        """Get detailed information about all registered parsers"""
        info = {}
        
        for doc_type, parser in self.parsers.items():
            try:
                field_patterns = parser._get_field_patterns()
                validation_rules = parser._get_validation_rules()
                
                info[doc_type.value] = {
                    "available": True,
                    "class_name": parser.__class__.__name__,
                    "supported_fields": list(field_patterns.keys()),
                    "validation_rules": list(validation_rules.keys()),
                    "field_pattern_count": {field: len(patterns) for field, patterns in field_patterns.items()},
                    "parser_type": "intelligent" if hasattr(parser, '_validate_with_name') else "standard"
                }
            except Exception as e:
                info[doc_type.value] = {
                    "available": False,
                    "error": str(e),
                    "class_name": parser.__class__.__name__
                }
        
        return info
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_parsers = len(self.parsers)
        available_types = [dt.value for dt in self.get_available_parsers()]
        
        return {
            "total_parsers": total_parsers,
            "available_document_types": available_types,
            "registry_health": "healthy" if total_parsers > 0 else "no_parsers",
            "coverage": {
                "aadhaar": DocumentType.AADHAAR in self.parsers,
                "pan": DocumentType.PAN in self.parsers,
                "driving_license": DocumentType.DRIVING_LICENSE in self.parsers,
                "voter_id": DocumentType.VOTER_ID in self.parsers
            }
        }


# Global parser registry instance
_parser_registry: Optional[ParserRegistry] = None


def get_parser_registry() -> ParserRegistry:
    """
    Get global parser registry instance (singleton pattern)
    
    Returns:
        Global ParserRegistry instance
    """
    global _parser_registry
    if _parser_registry is None:
        _parser_registry = ParserRegistry()
        logger.debug("Created new parser registry instance")
    return _parser_registry


def parse_document(doc_type: DocumentType, text: str, image_data: Any = None) -> ParseResult:
    """
    Convenience function for parsing documents
    
    Args:
        doc_type: Type of document to parse
        text: OCR extracted text or raw text
        image_data: Optional image data
        
    Returns:
        ParseResult with extracted fields and metadata
    """
    registry = get_parser_registry()
    return registry.parse_document(doc_type, text, image_data)


def get_available_parsers() -> List[DocumentType]:
    """Get list of available document types that can be parsed"""
    registry = get_parser_registry()
    return registry.get_available_parsers()


def has_parser_for(doc_type: DocumentType) -> bool:
    """Check if parser is available for specific document type"""
    registry = get_parser_registry()
    return registry.has_parser(doc_type)


# Test function
def test_parser_registry():
    """Test the parser registry with sample documents"""
    registry = get_parser_registry()
    
    # Test registry info
    info = registry.get_parser_info()
    stats = registry.get_registry_stats()
    
    # Test Aadhaar parsing if available
    aadhaar_result = None
    if has_parser_for(DocumentType.AADHAAR):
        sample_aadhaar = """
        GOVERNMENT OF INDIA
        AADHAAR
        Name: John Doe
        DOB: 15/08/1990
        Gender: MALE
        2314 5678 9012
        Address: 123 Main St, Mumbai, Maharashtra, 400001
        """
        aadhaar_result = parse_document(DocumentType.AADHAAR, sample_aadhaar)
    
    # Test PAN parsing if available  
    pan_result = None
    if has_parser_for(DocumentType.PAN):
        sample_pan = """
        INCOME TAX DEPARTMENT
        Permanent Account Number Card
        ABCDE1234F
        Name: JOHN DOE
        Father's Name: JAMES DOE
        Date of Birth: 15/08/1990
        """
        pan_result = parse_document(DocumentType.PAN, sample_pan)
    
    return {
        "registry_stats": stats,
        "parser_info": info,
        "aadhaar_test": {
            "available": aadhaar_result is not None,
            "confidence": aadhaar_result.confidence_score if aadhaar_result else 0,
            "fields_count": len(aadhaar_result.fields.__dict__) if aadhaar_result and aadhaar_result.fields else 0
        },
        "pan_test": {
            "available": pan_result is not None,
            "confidence": pan_result.confidence_score if pan_result else 0,
            "fields_count": len(pan_result.fields.__dict__) if pan_result and pan_result.fields else 0
        }
    }


if __name__ == "__main__":
    # Run comprehensive test if file is executed directly
    test_results = test_parser_registry()
    print("Parser registry test results:")
    for key, value in test_results.items():
        print(f"  {key}: {value}")
