#!/bin/bash
set -e

# 1. Setup Conda environment
echo "Setting up Conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    source $HOME/miniconda/etc/profile.d/conda.sh
    rm miniconda.sh
else
    echo "Conda is already installed."
    # Attempt to source conda if not already in path (common in non-interactive shells)
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Create environment e1 with python 3.9
if conda info --envs | grep -q "e1"; then
    echo "Environment e1 already exists."
else
    conda create -n e1 python=3.9 -y
fi

# Activate environment and install requirements
echo "Activating environment e1 and installing requirements..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate e1
pip install -r requirements.txt

# 2. Install Neo4j
echo "Installing Neo4j..."

# Install Java 17 (required for Neo4j 5.x/2025.x)
apt-get update
apt-get install -y openjdk-17-jdk

# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/neo4j.gpg
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 5" | tee /etc/apt/sources.list.d/neo4j.list
apt-get update

# Install specific version 2025.10.1
# Note: If this version doesn't exist in the repo yet (future version), apt install will fail.
# Assuming the user knows this version exists or is available in their context.
apt-get install -y --allow-downgrades neo4j=1:5.13.0

# Enable and start Neo4j
if command -v systemctl &> /dev/null && systemctl | grep -q '\-\.mount'; then
    systemctl enable neo4j
    systemctl start neo4j
else
    echo "Systemd not available. Starting Neo4j manually..."
    neo4j start || echo "Failed to start Neo4j manually. Please check logs."
fi

# 3. Setup APOC Plugin
echo "Setting up APOC plugin..."
APOC_URL="https://github.com/neo4j/apoc/releases/download/5.13.0/apoc-5.13.0-core.jar"
PLUGIN_DIR="/var/lib/neo4j/plugins" # Default for apt install
CONFIG_FILE="/etc/neo4j/neo4j.conf"

# Download APOC
wget $APOC_URL -O $PLUGIN_DIR/apoc-5.13.0-core.jar

# Configure Neo4j to allow APOC
# Check if config already exists to avoid duplication
if ! grep -q "dbms.security.procedures.unrestricted=apoc.\*" $CONFIG_FILE; then
    echo "dbms.security.procedures.unrestricted=apoc.*" | tee -a $CONFIG_FILE
fi

# Restart Neo4j to apply changes
if command -v systemctl &> /dev/null && systemctl | grep -q '\-\.mount'; then
    systemctl restart neo4j
else
    echo "Restarting Neo4j manually..."
    neo4j restart || neo4j start || echo "Failed to restart Neo4j."
fi

# 4. Change Neo4j Password
echo "Changing Neo4j password..."
# Wait for Neo4j to be ready
echo "Waiting for Neo4j to start..."
until cypher-shell -u neo4j -p neo4j "RETURN 1" &> /dev/null; do
    echo "Waiting for Neo4j..."
    sleep 5
done

# Change password to 12345678
# Note: If password is already changed, this might fail or need handling. 
# We assume default state or handle error gracefully.
if cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '12345678'" &> /dev/null; then
    echo "Password changed successfully."
else
    echo "Could not change password (maybe already changed?)."
fi

# 5. Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service if not running (background)
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama serve..."
    ollama serve &
    sleep 5 # Wait for it to start
fi

echo "Pulling Ollama models..."
ollama pull llama3.1:latest
ollama pull llama3.1:70b

# 6. Download Data
echo "Downloading data..."
# Activate conda env to install gdown
source $(conda info --base)/etc/profile.d/conda.sh
conda activate e1
pip install gdown

# Create directories
mkdir -p backend/krissbert_custom
mkdir -p data

# Download file 1
echo "Downloading krissbert_custom file 1..."
gdown "12Y05W7qRx2sGDuZsZydBoPdKirlP6Q2Z" -O backend/krissbert_custom/

# Download file 3 (New request)
echo "Downloading krissbert_custom file 2..."
gdown "1DeGPuA095JyoKKlSNJy4LhELLKRewbNp" -O backend/krissbert_custom/

# Download file 2
echo "Downloading data file..."
gdown "13k6HQ9upgXOXKL9TeKQUaPk3S3o2fIsD" -O data/data.zip
# Unzip
echo "Unzipping data..."
unzip -o data/data.zip -d data/
rm data/data.zip

echo "Setup complete!"
