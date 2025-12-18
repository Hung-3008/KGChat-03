#!/bin/bash
# Start additional Ollama instances on ports 11435 and 11436
# Port 11434 is assumed to be running via system service

# Check if 11434 is up
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "Starting default Ollama on 11434..."
    OLLAMA_HOST=127.0.0.1:11434 ollama serve > ollama_11434.log 2>&1 &
else
    echo "Ollama on 11434 is already running."
fi

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

start_ollama 11435
start_ollama 11436

echo "Waiting for services to start..."
sleep 5

# Verify
echo "Verifying ports:"
for port in 11434 11435 11436; do
    if curl -s http://localhost:$port/api/tags >/dev/null; then
        echo "Port $port: UP"
    else
        echo "Port $port: DOWN"
    fi
done
