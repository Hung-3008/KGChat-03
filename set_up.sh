#!/bin/bash

set -e  # Exit on error

echo "======================================"
echo "Starting setup script..."
echo "======================================"

# 1. Install and start Ollama (background mode)
echo ""
echo "Step 1: Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed successfully"
else
    echo "Ollama is already installed"
fi

# Start Ollama service in background
echo "Starting Ollama service..."
if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
    # If systemd is available and running
    systemctl enable ollama
    systemctl start ollama
    echo "Ollama service started via systemd"
else
    # If systemd is not available, run as background process
    nohup ollama serve > ollama.log 2>&1 &
    echo "Ollama started in background mode"
fi
sleep 3
echo "Ollama is running"

# 2. Install and start Qdrant service
echo ""
echo "Step 2: Installing Qdrant..."
QDRANT_VERSION="v1.16.3"

# Download Qdrant if not already present
if [ ! -f "./qdrant" ]; then
    wget https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
    tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
    rm qdrant-x86_64-unknown-linux-gnu.tar.gz
    echo "Qdrant downloaded and extracted"
else
    echo "Qdrant binary already exists"
fi

# Prepare storage directories
mkdir -p qdrant_storage
mkdir -p qdrant_snapshots

# Create symlinks if they don't exist
if [ ! -L "storage" ]; then
    ln -sf qdrant_storage storage
fi
if [ ! -L "snapshots" ]; then
    ln -sf qdrant_snapshots snapshots
fi

# Stop existing qdrant if running
pkill qdrant || true
sleep 2

echo "Starting Qdrant service..."
nohup ./qdrant > qdrant.log 2>&1 &
sleep 3
echo "Qdrant service started in background"

# 3. Install latest version of rclone
echo ""
echo "Step 3: Installing rclone (latest version)..."
if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | bash
    echo "rclone installed successfully"
else
    # Update to latest version
    echo "rclone is already installed, updating to latest version..."
    rclone selfupdate || curl https://rclone.org/install.sh | bash
fi
rclone version
echo "rclone is ready"

# 4. Install requirements
echo ""
echo "Step 4: Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Requirements installed successfully"
else
    echo "Warning: requirements.txt not found"
fi

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "Service status:"
echo "- Ollama: Running in background"
echo "- Qdrant: Running in background (check qdrant.log for logs)"
echo "- rclone: Installed and ready"
echo "- Python requirements: Installed"
echo ""


# gdowm: snapshot, model.bin, data 
# config rclone 
# rclone backend copyid may05: 1Fq0ev90cRZi3_RuKFL-ISJkjT7ifYRId snapshot.zip -P
# https://drive.google.com/file/d/13kwWOY7j-hg9t9DOURH8wjJKdUi1EONC/view?usp=sharing
# gdown 1XLTip64QcWuJYmGv7imoApbBXwxxb7JK
# https://drive.google.com/file/d/1PJRVZHrHzCUXqYS-Tc02HdcTycqwN6zd/view?usp=sharing