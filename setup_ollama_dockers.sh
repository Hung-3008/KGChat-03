#!/bin/bash

# Define common variables
OLLAMA_IMAGE="ollama/ollama:latest"
VOLUME_NAME="ollama_data"

# Create volume if it doesn't exist
if ! docker volume inspect $VOLUME_NAME > /dev/null 2>&1; then
    docker volume create $VOLUME_NAME
    echo "Created volume: $VOLUME_NAME"
fi

# Function to run ollama container
run_ollama() {
    local INSTANCE_NAME=$1
    local PORT=$2
    
    # Check if container exists
    if [ "$(docker ps -aq -f name=^/${INSTANCE_NAME}$)" ]; then
        echo "Removing existing container: $INSTANCE_NAME"
        docker rm -f $INSTANCE_NAME
    fi
    
    echo "Starting $INSTANCE_NAME on port $PORT..."
    docker run -d --gpus=all \
        -v $VOLUME_NAME:/root/.ollama \
        -p $PORT:11434 \
        --name $INSTANCE_NAME \
        --restart always \
        $OLLAMA_IMAGE
        
    echo "Started $INSTANCE_NAME"
}

# Run 3 instances
run_ollama "ollama-1" 11434
run_ollama "ollama-2" 11435
run_ollama "ollama-3" 11436

echo "All Ollama instances are running."
echo "Please ensure you pull the model in one instance (it's shared volume) if not present."
echo "Command: docker exec -it ollama-1 ollama pull llama3.2:3b"
