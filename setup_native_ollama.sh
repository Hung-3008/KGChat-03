# Start additional Ollama instances on ports 11435-11438
# Port 11434 is assumed to be running via system service, but we check it too.

# Configuration
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1 # Ensure we don't load too many models per instance if VRAM is tight

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

# Start instances
start_ollama 11434
start_ollama 11435
start_ollama 11436
start_ollama 11437
start_ollama 11438

echo "Waiting for services to start..."
sleep 5

# Verify
echo "Verifying ports:"
for port in 11434 11435 11436 11437 11438; do
    if curl -s http://localhost:$port/api/tags >/dev/null; then
        echo "Port $port: UP"
    else
        echo "Port $port: DOWN"
    fi
done
