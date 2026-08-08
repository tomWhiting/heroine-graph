#!/bin/bash
# GraphMother - WASM Build Script
#
# Usage: ./build.sh [--release|--dev] [--simd]
#
# Options:
#   --release   Build optimized release version (default)
#   --dev       Build debug version with faster compile times
#   --simd      Enable WASM SIMD for vectorized operations
#   --clean     Remove previous build artifacts before building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
PROFILE="release"
SIMD_ENABLED=false
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            PROFILE="release"
            shift
            ;;
        --dev)
            PROFILE="dev"
            shift
            ;;
        --simd)
            SIMD_ENABLED=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./build.sh [--release|--dev] [--simd] [--clean]"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning previous build artifacts..."
    rm -rf pkg/
    cargo clean -p graphmother-wasm
fi

# Check for required tools
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Install with: cargo install wasm-pack"
    exit 1
fi

# Append SIMD flags to any RUSTFLAGS already set by the environment or
# .cargo/config.toml. Never export RUSTFLAGS when we have nothing to add:
# an empty export would clobber the ambient value.
if [ "$SIMD_ENABLED" = true ]; then
    echo "Building with WASM SIMD enabled..."
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
fi

# Build
echo "Building graphmother-wasm (profile: $PROFILE)..."

if [ "$PROFILE" = "release" ]; then
    wasm-pack build --target web --release --out-dir pkg
else
    wasm-pack build --target web --dev --out-dir pkg
fi

# Post-build verification
if [ -f pkg/graphmother_wasm.js ]; then
    echo ""
    echo "Build successful!"

    # The npm manifest for this package is packages/wasm/package.json, which is
    # checked in and edited by hand. wasm-pack also writes a pkg/package.json;
    # nothing reads it, and it is not what gets published.

    echo "Output files:"
    ls -lh pkg/
    echo ""
    echo "WASM size: $(ls -lh pkg/*.wasm | awk '{print $5}')"
else
    echo "Error: Build failed - output files not found"
    exit 1
fi
