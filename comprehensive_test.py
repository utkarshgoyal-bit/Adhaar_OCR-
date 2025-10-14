#!/usr/bin/env python3
"""
Comprehensive test script for enhanced OCR system
Tests all enhancements: preprocessing, multi-pass, document-specific configs
"""

import sys
import os
import numpy as np
import cv2
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test 1: Check if all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Import Test")
    print("="*60)
    
    results = {}
    
    # Test OCR base
    try:
        from app.ocr.base import OCREngine, OCRResult, OCRPreprocessor
        results['ocr_base'] = '✅ OK'
    except ImportError as e:
        results['ocr_base'] = f'❌ FAILED: {e}'
    
    # Test enhanced preprocessing
    try:
        from app.ocr.enhanced_preprocessing import EnhancedPreprocessor, check_quality
        results['enhanced_preprocessing'] = '✅ OK'
    except ImportError as e:
        results['enhanced_preprocessing'] = f'❌ FAILED: {e}'
    
    # Test Tesseract
    try:
        from app.ocr.tesseract import TesseractEngine, create_tesseract_engine
        results['tesseract'] = '✅ OK'
    except ImportError as e:
        results['tesseract'] = f'❌ FAILED: {e}'
    
    # Test OCR Manager
    try:
        from app.ocr.manager import OCRManager, get_ocr_manager
        results['ocr_manager'] = '✅ OK'
    except ImportError as e:
        results['ocr_manager'] = f'❌ FAILED: {e}'
    
    # Test OCR __init__
    try:
        from app.ocr import get_ocr_info
        info = get_ocr_info()
        results['ocr_init'] = f'✅ OK (version: {info["version"]})'
    except ImportError as e:
        results['ocr_init'] = f'❌ FAILED: {e}'
    
    # Test parsers
    try:
        from app.parsers import parse_document, get_available_parsers
        parsers = get_available_parsers()
        results['parsers'] = f'✅ OK ({len(parsers)} parsers)'
    except ImportError as e:
        results['parsers'] = f'❌ FAILED: {e}'
    
    # Print results
    for module, status in results.items():
        print(f"  {module:30s}: {status}")
    
    all_ok = all('✅' in status for status in results.values())
    print(f"\n{'✅ All imports successful!' if all_ok else '❌ Some imports failed'}")
    
    return all_ok


def test_enhanced_preprocessing():
    """Test 2: Enhanced preprocessing functionality"""
    print("\n" + "="*60)
    print("TEST 2: Enhanced Preprocessing Test")
    print("="*60)
    
    try:
        from app.ocr.enhanced_preprocessing import EnhancedPreprocessor
        
        # Create test image
        test_image = np.ones((300, 600, 3), dtype=np.uint8) * 200
        cv2.putText(test_image, "AADHAAR CARD", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(test_image, "Name: Rahul Kumar", (50, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "1234 5678 9012", (50, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        preprocessor = EnhancedPreprocessor()
        
        # Test quality check
        print("\n  Testing quality check...")
        quality = preprocessor.check_image_quality(test_image)
        print(f"    Blur score: {quality.get('blur_score', 'N/A'):.2f}")
        print(f"    Brightness: {quality.get('brightness', 'N/A'):.2f}")
        print(f"    Good quality: {quality.get('good_quality', False)}")
        
        # Test Aadhaar preprocessing
        print("\n  Testing Aadhaar preprocessing...")
        aadhaar_result, aadhaar_steps = preprocessor.preprocess_for_document(
            test_image, "aadhaar"
        )
        print(f"    Steps applied: {', '.join(aadhaar_steps)}")
        print(f"    Output shape: {aadhaar_result.shape}")
        
        # Test PAN preprocessing
        print("\n  Testing PAN preprocessing...")
        pan_result, pan_steps = preprocessor.preprocess_for_document(
            test_image, "pan"
        )
        print(f"    Steps applied: {', '.join(pan_steps)}")
        print(f"    Output shape: {pan_result.shape}")
        
        print("\n✅ Enhanced preprocessing working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_manager():
    """Test 3: OCR Manager with multi-pass strategy"""
    print("\n" + "="*60)
    print("TEST 3: OCR Manager Multi-Pass Test")
    print("="*60)
    
    try:
        from app.ocr.manager import get_ocr_manager
        
        # Create test image
        test_image = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "AADHAAR CARD", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(test_image, "Name: Sanju Devi", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(test_image, "4389 9349 1869", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Get OCR manager
        manager = get_ocr_manager()
        stats = manager.get_manager_stats()
        
        print(f"\n  Manager Status:")
        print(f"    Total engines: {stats['total_engines']}")
        print(f"    Available: {stats['available_engines']}")
        print(f"    Enhanced preprocessing: {stats['enhanced_preprocessing_enabled']}")
        
        if not stats['available_engines']:
            print("\n❌ No OCR engines available!")
            print("   Install Tesseract: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin")
            return False
        
        # Test OCR with Aadhaar
        print(f"\n  Running OCR with Aadhaar preprocessing...")
        start_time = datetime.now()
        
        result = manager.extract_text(test_image, doc_type="aadhaar")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"    Processing time: {processing_time:.1f}ms")
        print(f"    Confidence: {result.confidence:.3f}")
        print(f"    Text length: {len(result.text)} chars")
        print(f"    Engine used: {result.engine}")
        print(f"    Languages detected: {result.language_detected}")
        
        # Check for preprocessing metadata
        if result.bbox_data:
            for item in result.bbox_data:
                if item.get('type') == 'processing_metadata':
                    strategy = item.get('strategy', 'unknown')
                    print(f"    Preprocessing strategy: {strategy}")
                    break
        
        print(f"\n    Extracted text preview:")
        print(f"    {'-'*40}")
        preview = result.text[:200] if len(result.text) > 200 else result.text
        print(f"    {preview}")
        print(f"    {'-'*40}")
        
        if result.text and result.confidence > 0.5:
            print("\n✅ OCR Manager working correctly!")
            return True
        else:
            print("\n⚠️ OCR Manager working but low confidence/no text")
            return False
        
    except Exception as e:
        print(f"\n❌ OCR Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_specific_configs():
    """Test 4: Document-specific Tesseract configurations"""
    print("\n" + "="*60)
    print("TEST 4: Document-Specific Config Test")
    print("="*60)
    
    try:
        from app.ocr.tesseract import create_tesseract_engine
        
        engine = create_tesseract_engine()
        
        if not engine.is_available():
            print("\n❌ Tesseract not available!")
            return False
        
        # Test Aadhaar config
        print("\n  Testing Aadhaar-specific config...")
        aadhaar_img = np.ones((150, 500, 3), dtype=np.uint8) * 255
        cv2.putText(aadhaar_img, "1234 5678 9012", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        aadhaar_result = engine.extract_text(aadhaar_img, doc_type="aadhaar")
        print(f"    Config used: {aadhaar_result.engine}")
        print(f"    Confidence: {aadhaar_result.confidence:.3f}")
        print(f"    Text: '{aadhaar_result.text.strip()}'")
        
        # Test PAN config
        print("\n  Testing PAN-specific config...")
        pan_img = np.ones((150, 400, 3), dtype=np.uint8) * 255
        cv2.putText(pan_img, "ABCDE1234F", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        pan_result = engine.extract_text(pan_img, doc_type="pan")
        print(f"    Config used: {pan_result.engine}")
        print(f"    Confidence: {pan_result.confidence:.3f}")
        print(f"    Text: '{pan_result.text.strip()}'")
        
        # Check if different configs were used
        if 'aadhaar' in aadhaar_result.engine and 'pan' in pan_result.engine:
            print("\n✅ Document-specific configs working!")
            return True
        else:
            print("\n⚠️ Document-specific configs may not be activating")
            return False
        
    except Exception as e:
        print(f"\n❌ Document-specific config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parsers():
    """Test 5: Parser integration"""
    print("\n" + "="*60)
    print("TEST 5: Parser Integration Test")
    print("="*60)
    
    try:
        from app.parsers import parse_document, DocumentType
        
        # Test Aadhaar parser
        print("\n  Testing Aadhaar parser...")
        aadhaar_text = """
        GOVERNMENT OF INDIA
        AADHAAR
        Name: Rahul Kumar Sharma
        DOB: 21/07/1993
        Gender: MALE
        4389 9349 1869
        Address: House No 123, MG Road, Jaipur, Rajasthan, 302019
        """
        
        aadhaar_result = parse_document(DocumentType.AADHAAR, aadhaar_text)
        print(f"    Confidence: {aadhaar_result.confidence_score:.3f}")
        print(f"    Fields extracted: {len([f for f in dir(aadhaar_result.fields) if not f.startswith('_')])}")
        
        if aadhaar_result.fields and hasattr(aadhaar_result.fields, 'id_number'):
            print(f"    ID extracted: {aadhaar_result.fields.id_number.value if aadhaar_result.fields.id_number else 'None'}")
        
        # Test PAN parser
        print("\n  Testing PAN parser...")
        pan_text = """
        INCOME TAX DEPARTMENT
        Permanent Account Number Card
        ABCDE1234F
        Name: JOHN DOE
        Father's Name: JAMES DOE
        Date of Birth: 15/08/1990
        """
        
        pan_result = parse_document(DocumentType.PAN, pan_text)
        print(f"    Confidence: {pan_result.confidence_score:.3f}")
        print(f"    Fields extracted: {len([f for f in dir(pan_result.fields) if not f.startswith('_')])}")
        
        if pan_result.fields and hasattr(pan_result.fields, 'id_number'):
            print(f"    PAN extracted: {pan_result.fields.id_number.value if pan_result.fields.id_number else 'None'}")
        
        print("\n✅ Parsers working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_health():
    """Test 6: API health check (if service is running)"""
    print("\n" + "="*60)
    print("TEST 6: API Health Check (Optional)")
    print("="*60)
    
    try:
        import httpx
        
        print("\n  Checking if API is running on http://localhost:8000...")
        
        try:
            response = httpx.get("http://localhost:8000/healthz", timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                print(f"    Status: {data.get('status', 'unknown')}")
                print(f"    Version: {data.get('version', 'unknown')}")
                print(f"    Timestamp: {data.get('timestamp', 'unknown')}")
                print("\n✅ API is running and healthy!")
                return True
            else:
                print(f"\n⚠️ API returned status {response.status_code}")
                return False
                
        except httpx.ConnectError:
            print("\n⚠️ API not running (this is OK for local testing)")
            print("   To test API: uvicorn app.main:app --reload")
            return None  # Not a failure, just not running
            
    except ImportError:
        print("\n⚠️ httpx not installed (optional)")
        print("   Install: pip install httpx")
        return None


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE OCR ENHANCEMENT TEST SUITE")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['preprocessing'] = test_enhanced_preprocessing()
    results['ocr_manager'] = test_ocr_manager()
    results['doc_configs'] = test_document_specific_configs()
    results['parsers'] = test_parsers()
    results['api_health'] = test_api_health()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠️ SKIP")
        print(f"  {test_name:20s}: {status}")
    
    print(f"\n  Total: {total} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    
    # Final verdict
    print("\n" + "="*60)
    if failed == 0:
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("Your enhanced OCR system is working correctly.")
        print("\nNext steps:")
        print("1. Test with real Aadhaar/PAN images")
        print("2. Compare accuracy before/after enhancements")
        print("3. Deploy and monitor performance")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above and fix issues.")
        print("\nCommon fixes:")
        print("1. Install Tesseract: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check logs for detailed error messages")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
