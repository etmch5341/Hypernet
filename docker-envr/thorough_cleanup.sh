#!/bin/bash
# thorough_cleanup.sh - Thoroughly clean ALL build artifacts from esy-osmfilter

set -e

echo "============================================"
echo "Thorough Cleanup of esy-osmfilter Directory"
echo "============================================"
echo ""

if [ ! -d "esy-osmfilter" ]; then
    echo "❌ Error: esy-osmfilter directory not found"
    echo "Make sure you run this script from the project root"
    exit 1
fi

cd esy-osmfilter

echo "🔍 Searching for .egg-info directories..."
EGG_DIRS=$(find . -type d -name "*.egg-info" 2>/dev/null)

if [ -z "$EGG_DIRS" ]; then
    echo "✓ No .egg-info directories found"
else
    echo "Found the following .egg-info directories:"
    echo "$EGG_DIRS"
    echo ""
    echo "🗑️  Removing .egg-info directories..."
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    echo "✓ Removed .egg-info directories"
fi

echo ""
echo "🗑️  Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Removed __pycache__ directories"

echo ""
echo "🗑️  Removing .pyc and .pyo files..."
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
echo "✓ Removed compiled Python files"

echo ""
echo "🗑️  Removing build, dist, and .eggs directories..."
rm -rf build/ dist/ .eggs/ 2>/dev/null || true
echo "✓ Removed build directories"

echo ""
echo "🗑️  Removing .pytest_cache and .tox..."
rm -rf .pytest_cache/ .tox/ 2>/dev/null || true
echo "✓ Removed test cache directories"

cd ..

echo ""
echo "============================================"
echo "✅ Cleanup Complete!"
echo "============================================"
echo ""
echo "Verification - searching for any remaining .egg-info:"
find esy-osmfilter -type d -name "*.egg-info" 2>/dev/null || echo "None found ✓"
echo ""
echo "Next steps:"
echo "1. Clean Docker cache: docker-compose down && docker system prune -f"
echo "2. Rebuild container: docker-compose build --no-cache"