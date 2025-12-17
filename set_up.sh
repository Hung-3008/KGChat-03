#sudo mkdir -p /etc/systemd/system/ollama.service.d
echo '[Service]
Environment="OLLAMA_NUM_PARALLEL=32"
Environment="OLLAMA_MAX_LOADED_MODELS=2"' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama | grep Environment