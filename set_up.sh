# #sudo mkdir -p /etc/systemd/system/ollama.service.d
# echo '[Service]
# Environment="OLLAMA_NUM_PARALLEL=42"
# Environment="OLLAMA_MAX_LOADED_MODELS=2"' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
# sudo systemctl daemon-reload
# sudo systemctl restart ollama
# systemctl show ollama | grep Environment
# export GOOGLE_APPLICATION_CREDENTIALS="/home/hung/.gcp-keys/vertex-ai-key.json"


# Install & Run Qdrant (Binary mode)
echo "Installing Qdrant..."
QDRANT_VERSION="latest" # Or latest
wget https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Prepare storage directories to match docker-compose logic
mkdir -p qdrant_storage
mkdir -p qdrant_snapshots

# Stop existing qdrant if running
pkill qdrant || true

#rm -rf storage
ln -s qdrant_storage storage
#rm -rf snapshots
ln -s qdrant_snapshots snapshots

echo "Starting Qdrant..."
nohup ./qdrant > qdrant.log 2>&1 &


# requirements 
pip install -r requirements.txt

# apt-get install -y rsync
# rsync -P -r hung@100.88.234.38:/media/hung/data1/codes/projects/FHC/backups/snapshot/ .


rclone backend copyid kgchat: 1Fq0ev90cRZi3_RuKFL-ISJkjT7ifYRId . -P