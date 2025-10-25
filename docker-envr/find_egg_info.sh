#!/bin/bash
# find_egg_info.sh - Diagnostic script to find all .egg-info directories

echo "Searching for .egg-info directories in esy-osmfilter..."
echo "======================================================="
echo ""

if [ ! -d "esy-osmfilter" ]; then
    echo "Error: esy-osmfilter directory not found"
    exit 1
fi

cd esy-osmfilter

echo "All .egg-info directories found:"
find . -type d -name "*.egg-info" -ls 2>/dev/null

echo ""
echo "======================================================="
echo "Count: $(find . -type d -name '*.egg-info' 2>/dev/null | wc -l) directories found"
echo ""

if [ $(find . -type d -name "*.egg-info" 2>/dev/null | wc -l) -gt 0 ]; then
    echo "⚠️  Found .egg-info directories! Run thorough_cleanup.sh to remove them."
    echo ""
    echo "Locations:"
    find . -type d -name "*.egg-info" 2>/dev/null
else
    echo "✓ No .egg-info directories found - you're good to build!"
fi