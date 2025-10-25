#!/bin/bash
# cleanup_esy_osmfilter.sh
# Run this script to clean up build artifacts from your esy-osmfilter directory

echo "Cleaning up esy-osmfilter build artifacts..."

cd esy-osmfilter || { echo "Error: esy-osmfilter directory not found"; exit 1; }

# Remove .egg-info directories
echo "Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
echo "Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove build and dist directories
echo "Removing build/dist directories..."
rm -rf build/ dist/ .eggs/ 2>/dev/null || true

echo "✓ Cleanup complete!"
echo ""
echo "You can now rebuild your Docker container:"
echo "  docker-compose build --no-cache"