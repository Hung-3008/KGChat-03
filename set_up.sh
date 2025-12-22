# #sudo mkdir -p /etc/systemd/system/ollama.service.d
# echo '[Service]
# Environment="OLLAMA_NUM_PARALLEL=42"
# Environment="OLLAMA_MAX_LOADED_MODELS=2"' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
# sudo systemctl daemon-reload
# sudo systemctl restart ollama
# systemctl show ollama | grep Environment
# export GOOGLE_APPLICATION_CREDENTIALS="/home/hung/.gcp-keys/vertex-ai-key.json"


# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker on Ubuntu..."
    
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add the repository to Apt sources:
    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Enable and start Docker service
    sudo systemctl enable docker
    sudo systemctl start docker

    # Add current user to docker group
    sudo usermod -aG docker $USER
    echo "Docker installed successfully. Please log out and back in for group changes to take effect."
else
    echo "Docker is already installed."
fi


# Download data
pip install gdown

gdown 1j7fHaB-Oe0vZsMs3fTqD6Rr9vI50Li7r # part 2
gdown 12ZTo0oLpkFucgdxDrhEWnTEOGnAPxAdV # model.bin 
gdown 1PJRVZHrHzCUXqYS-Tc02HdcTycqwN6zd # snapshot 

# qdrant 
docker pull qdrant/qdrant
docker compose up -d

# requirements 
pip install -r requirements.txt

