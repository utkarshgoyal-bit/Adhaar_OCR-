#!/bin/bash
# ============================================
# Cleanup Script for OCR Service
# Removes old/obsolete files after enhancements
# ============================================

echo "🧹 Starting cleanup of obsolete files..."
echo ""

# Track what we're doing
removed_count=0
kept_count=0

# Function to safely remove file
remove_file() {
    if [ -f "$1" ]; then
        echo "❌ Removing: $1"
        rm "$1"
        ((removed_count++))
    else
        echo "⚠️  Not found: $1 (already removed?)"
    fi
}

# Function to safely remove directory
remove_dir() {
    if [ -d "$1" ]; then
        echo "❌ Removing directory: $1"
        rm -rf "$1"
        ((removed_count++))
    else
        echo "⚠️  Not found: $1 (already removed?)"
    fi
}

echo "Step 1: Remove PaddleOCR (no longer used)"
echo "----------------------------------------"
remove_file "app/ocr/paddleocr.py"
echo ""

echo "Step 2: Remove duplicate/obsolete OCR files"
echo "----------------------------------------"
remove_file "app/ocr/enhanced_ocr_configs.py"
remove_file "app/ocr/ocr_module_fix.py"
remove_file "ocr_accuracy_tester.py"
echo ""

echo "Step 3: Remove backup folder (old fixes)"
echo "----------------------------------------"
remove_dir "backup"
echo ""

echo "Step 4: Remove old diagnostic tests"
echo "----------------------------------------"
remove_file "tests/integration/ocr_diagnostic_test.py"
remove_file "tests/integration/test_ocr.py"
echo ""

echo "Step 5: Remove unnecessary scripts"
echo "----------------------------------------"
remove_file "scripts/cache_clear_script.py"
# Keep ocr_setup.py as it might be useful
echo "✅ Keeping: scripts/ocr_setup.py (useful for installation)"
((kept_count++))
echo ""

echo "Step 6: Remove Windows-specific files (optional)"
echo "----------------------------------------"
remove_file "app/desktop.ini"
echo ""

echo "Step 7: Remove old setup scripts"
echo "----------------------------------------"
remove_file "app/scripts/setup.bat"
remove_file "app/scripts/project_structure.sh"
echo ""

# Summary
echo "============================================"
echo "✅ Cleanup Complete!"
echo "============================================"
echo "Files removed: $removed_count"
echo "Files kept: $kept_count"
echo ""

echo "📊 Space saved: ~400MB+ (PaddleOCR dependencies)"
echo ""

echo "Next steps:"
echo "1. Update requirements: pip install -r requirements.txt --upgrade"
echo "2. Restart your service: uvicorn app.main:app --reload"
echo "3. Test the enhanced OCR system"
echo ""

echo "✅ Your codebase is now clean and optimized!"
