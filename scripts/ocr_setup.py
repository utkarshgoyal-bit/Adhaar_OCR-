#!/usr/bin/env python3
"""
OCR Installation and Setup Verification Script
Run this to diagnose and fix OCR issues in your document processing service
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Python packages...")
    
    packages_to_check = {
        'pytesseract': 'pytesseract',
        'paddleocr': 'paddleocr', 
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'fitz': 'PyMuPDF'
    }
    
    missing_packages = []
    
    for import_name, package_name in packages_to_check.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n🚀 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_tesseract_binary():
    """Check if Tesseract binary is installed and accessible"""
    print("\n🔍 Checking Tesseract binary...")
    
    # Common Tesseract paths
    common_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/homebrew/bin/tesseract',  # macOS with Homebrew
        'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  # Windows
        'tesseract'  # In PATH
    ]
    
    tesseract_found = False
    
    for path in common_paths:
        if os.path.exists(path) or run_command(f"{path} --version", f"Testing Tesseract at {path}"):
            print(f"✅ Tesseract found at: {path}")
            tesseract_found = True
            break
    
    if not tesseract_found:
        print("❌ Tesseract binary not found!")
        print("\n📥 Install Tesseract:")
        print("Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-hin")
        print("CentOS/RHEL: sudo yum install tesseract tesseract-langpack-hin")
        print("macOS: brew install tesseract tesseract-lang")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    return True

def check_tesseract_languages():
    """Check available Tesseract language packs"""
    print("\n🌐 Checking Tesseract languages...")
    
    if run_command("tesseract --list-langs", "Listing available languages"):
        print("✅ Language check completed")
        print("📝 For Indian documents, ensure you have: eng, hin")
        print("Install if missing: sudo apt install tesseract-ocr-hin")
        return True
    else:
        print("❌ Could not check languages")
        return False

def test_paddleocr():
    """Test PaddleOCR installation"""
    print("\n🏓 Testing PaddleOCR...")
    
    test_code = '''
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("✅ PaddleOCR initialized successfully")
except Exception as e:
    print(f"❌ PaddleOCR failed: {e}")
'''
    
    return run_command(f'python3 -c "{test_code}"', "Testing PaddleOCR initialization")

def create_ocr_test_script():
    """Create a test script to verify OCR functionality"""
    
    test_script = '''#!/usr/bin/env python3
"""
OCR Functionality Test Script
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use a better font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((20, 50), "Test OCR Text", fill='black', font=font)
    draw.text((20, 100), "Aadhaar: 1234 5678 9012", fill='black', font=font)
    draw.text((20, 150), "Name: Rahul Kumar", fill='black', font=font)
    
    return img

def test_tesseract():
    """Test Tesseract OCR"""
    print("\\n🔤 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Create test image
        img = create_test_image()
        
        # Test English
        text_eng = pytesseract.image_to_string(img, lang='eng')
        print(f"Tesseract (English): {text_eng.strip()}")
        
        # Test Hindi (if available)
        try:
            text_hin = pytesseract.image_to_string(img, lang='eng+hin')
            print(f"Tesseract (Eng+Hindi): {text_hin.strip()}")
        except:
            print("Hindi language pack not available for Tesseract")
        
        return True
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def test_paddleocr_actual():
    """Test actual PaddleOCR functionality"""
    print("\\n🏓 Testing PaddleOCR functionality...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Initialize OCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Create test image
        img = create_test_image()
        img_array = np.array(img)
        
        # Perform OCR
        result = ocr.ocr(img_array, cls=True)
        
        print("PaddleOCR Results:")
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                print(f"  Text: '{text}' (Confidence: {confidence:.2f})")
        else:
            print("  No text detected")
        
        return True
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 OCR Functionality Test")
    print("=" * 40)
    
    success = True
    success &= test_tesseract()
    success &= test_paddleocr_actual()
    
    if success:
        print("\\n✅ All OCR tests passed!")
    else:
        print("\\n❌ Some OCR tests failed. Check the errors above.")
'''
    
    with open('test_ocr.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_ocr.py', 0o755)
    print("\n📝 Created test_ocr.py - run this to test OCR functionality")

def main():
    """Main diagnosis and setup function"""
    print("🔍 OCR Setup and Diagnosis Tool")
    print("=" * 50)
    
    success = True
    
    # Step 1: Check Python packages
    success &= check_python_packages()
    
    # Step 2: Check Tesseract binary
    success &= check_tesseract_binary()
    
    # Step 3: Check Tesseract languages
    if success:
        check_tesseract_languages()
    
    # Step 4: Test PaddleOCR
    if success:
        test_paddleocr()
    
    # Step 5: Create test script
    create_ocr_test_script()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ OCR setup looks good!")
        print("🚀 Run 'python test_ocr.py' to test actual OCR functionality")
    else:
        print("❌ OCR setup has issues. Follow the instructions above to fix them.")
    
    print("\n🔧 Quick fixes:")
    print("1. Install packages: pip install pytesseract paddleocr opencv-python pillow PyMuPDF")
    print("2. Install Tesseract: sudo apt install tesseract-ocr tesseract-ocr-hin")
    print("3. Test with: python test_ocr.py")

if __name__ == "__main__":
    main()