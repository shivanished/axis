#!/bin/bash
# Build script for creating macOS app bundle

set -e

echo "Building Axis macOS app..."

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: PyInstaller is not installed."
    echo "Install it with: pip install pyinstaller"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist

# Build the app
echo "Building app bundle..."
pyinstaller Axis.spec --clean

# Check if build was successful
if [ -d "dist/Axis.app" ]; then
    echo "✓ Build successful!"
    echo "App bundle location: dist/Axis.app"
    echo ""
    echo "To run the app:"
    echo "  open 'dist/Axis.app'"
    echo ""
    echo "Note: You may need to grant camera and accessibility permissions"
    echo "in System Settings > Privacy & Security"
else
    echo "✗ Build failed!"
    exit 1
fi
