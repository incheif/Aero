#!/usr/bin/env bash
# AERO Headless Runner Utility Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "⚡ AERO: Agentic Environment for Robotic Operations"
echo "Launching headless simulation container..."
echo "=========================================================="

IMAGE_NAME="aero-ros2"

# Build image if not present
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image '$IMAGE_NAME'..."
    docker build -t $IMAGE_NAME -f "$SCRIPT_DIR/Dockerfile" "$WORKSPACE_ROOT"
fi

# Run container with volume mount and headless display
docker run -it --rm \
    --ipc=host \
    --net=host \
    -e DISPLAY="" \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e QT_QPA_PLATFORM=offscreen \
    -v "$WORKSPACE_ROOT":/workspace \
    -w /workspace \
    $IMAGE_NAME "$@"
