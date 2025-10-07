#!/usr/bin/env python3
"""
Clear Python module cache and restart API
"""

import os
import sys
import shutil
import subprocess
import time

def clear_python_cache():
    """Remove all Python cache files"""
    print("Clearing Python cache...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for d in dirs[:]:  # Use slice to avoid modification during iteration
            if d == '__pycache__':
                cache_path = os.path.join(root, d)
                print(f"Removing: {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
                dirs.remove(d)
    
    # Remove .pyc files
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.pyc'):
                pyc_path = os.path.join(root, f)
                print(f"Removing: {pyc_path}")
                try:
                    os.remove(pyc_path)
                except OSError:
                    pass
    
    print("Cache cleared successfully!")

def verify_ocr_imports():
    """Test if OCR imports work correctly"""
    print("\nTesting OCR imports...")
    
    try:
        # Clear any existing imports
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith('app.ocr')]
        for module in modules_to_clear:
            del sys.modules[module]
        
        # Test imports
        from app.ocr import get_ocr_manager, MANAGER_AVAILABLE
        print(f"✅ OCR imports successful")
        print(f"✅ Manager available: {MANAGER_AVAILABLE}")
        
        if MANAGER_AVAILABLE:
            manager = get_ocr_manager()
            stats = manager.get_manager_stats()
            print(f"✅ OCR engines available: {stats['available_engines']}")
            return True
        else:
            print("❌ OCR Manager not available")
            return False
        
    except Exception as e:
        print(f"❌ OCR import failed: {e}")
        return False

def main():
    print("Python Module Cache Cleaner")
    print("=" * 30)
    
    # Clear cache
    clear_python_cache()
    
    # Test imports
    ocr_ok = verify_ocr_imports()
    
    if ocr_ok:
        print("\n🎉 All imports working correctly!")
        print("Your API should now have access to the OCR system.")
        print("\nTo restart your API:")
        print("1. Stop the current API (Ctrl+C)")
        print("2. Run: uvicorn app.main:app --reload")
    else:
        print("\n⚠️ Import issues still exist")
        print("Check that your app/ocr/__init__.py has the proper exports")

if __name__ == "__main__":
    main()
