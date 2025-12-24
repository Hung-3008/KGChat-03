#!/bin/bash
# Start Ollama instances on ports defined in backend/configs/configs.yml

# Hardcoded Configuration
# You can manually change these values
BASE_PORT=11434
NUM_PORTS=12

echo "Configuration: Base Port=$BASE_PORT, Num Ports=$NUM_PORTS"

# Generate ports array
PORTS=()
for ((i=0; i<NUM_PORTS; i++)); do
    PORTS+=($((BASE_PORT + i)))
done

echo "Found ${#PORTS[@]} ports: ${PORTS[*]}"

# Configuration
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=5 # Ensure we don't load too many models per instance if VRAM is tight

# Function to start if not running
start_ollama() {
    PORT=$1
    if ! curl -s http://localhost:$PORT/api/tags >/dev/null; then
        echo "Starting Ollama on $PORT..."
        OLLAMA_HOST=127.0.0.1:$PORT ollama serve > ollama_$PORT.log 2>&1 &
    else
        echo "Ollama on $PORT is already running."
    fi
}

# Start instances for each port in config
for port in "${PORTS[@]}"; do
    start_ollama $port
    sleep 2 # Wait 2s to avoid race condition on GPU init
done

echo "Waiting for services to start..."
sleep 5

# Verify
echo "Verifying ports:"
for port in "${PORTS[@]}"; do
    if curl -s http://localhost:$port/api/tags >/dev/null; then
        echo "Port $port: UP"
    else
        echo "Port $port: DOWN"
    fi
done
