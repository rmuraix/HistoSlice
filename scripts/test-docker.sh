#!/bin/bash
# Docker validation script for HistoSlice
# This script tests the Docker build and basic functionality

set -e

echo "🐳 HistoSlice Docker Validation Script"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is installed"

# Build the Docker image
print_info "Building Docker image..."
if docker build -t histoslice-test:local .; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Test 1: Check if CLI is accessible
print_info "Testing CLI accessibility..."
if docker run --rm histoslice-test:local --help > /dev/null 2>&1; then
    print_success "CLI is accessible"
else
    print_error "CLI is not accessible"
    exit 1
fi

# Test 2: Verify Python API is available
print_info "Testing Python API..."
if docker run --rm histoslice-test:local python -c "from histoslice import SlideReader; print('OK')" | grep -q "OK"; then
    print_success "Python API is accessible"
else
    print_error "Python API is not accessible"
    exit 1
fi

# Test 3: Check UV is available
print_info "Testing UV availability..."
if docker run --rm histoslice-test:local uv --version > /dev/null 2>&1; then
    print_success "UV is available"
else
    print_error "UV is not available"
    exit 1
fi

# Test 4: Verify OpenCV is installed
print_info "Testing OpenCV..."
if docker run --rm histoslice-test:local python -c "import cv2; print('OK')" | grep -q "OK"; then
    print_success "OpenCV is installed"
else
    print_error "OpenCV is not installed"
    exit 1
fi

# Test 5: Check data directories
print_info "Testing data directories..."
if docker run --rm histoslice-test:local sh -c "test -d /data/input && test -d /data/output && echo 'OK'" | grep -q "OK"; then
    print_success "Data directories exist"
else
    print_error "Data directories do not exist"
    exit 1
fi

echo ""
print_success "All Docker validation tests passed!"
echo ""
print_info "You can now push the image or test with real data:"
echo "  docker run --rm -v \$(pwd)/slides:/data/input -v \$(pwd)/output:/data/output histoslice-test:local --input '/data/input/*.tiff' --output /data/output --width 512"
